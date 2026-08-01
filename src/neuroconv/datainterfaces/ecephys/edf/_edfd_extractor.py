"""
SpikeInterface recording extractor for EDF/EDF+ files, including discontinuous (``EDF+D``) ones.

The parsing itself lives in :mod:`._edfd_reader`, which depends only on NumPy. This module is where
SpikeInterface enters, so it must only ever be imported lazily — SpikeInterface is an optional
("ecephys") extra. Import it from inside a function, as
:meth:`~...edf.edfdatainterface.EDFRecordingInterface.get_extractor_class` does.

The classes are defined at module scope on purpose: SpikeInterface records an extractor's class by
its import path in ``to_dict()``, so classes built inside a factory function cannot be pickled or
reloaded, which breaks ``recording.save()`` and every ``n_jobs > 1`` path.
"""

import numpy as np
from pydantic import FilePath
from spikeinterface.core import BaseRecording, BaseRecordingSegment

from ._edfd_reader import (
    EDFHeader,
    _build_channel_layout,
    group_records_into_runs,
    read_edf_header,
    read_record_onsets_and_annotations,
)


class EDFDRecordingSegment(BaseRecordingSegment):
    """One contiguous run of EDF data records, read lazily from disk."""

    def __init__(
        self,
        *,
        file_path: str,
        header: EDFHeader,
        first_record_index: int,
        number_of_records: int,
        samples_per_record: int,
        sample_offsets_in_record: np.ndarray,
        sampling_frequency: float,
        t_start: float,
        log_transforms: dict | None = None,
        dtype: str = "int16",
    ):
        super().__init__(sampling_frequency=sampling_frequency, t_start=t_start)
        self._log_transforms = log_transforms or dict()
        self._dtype = dtype
        self._file_path = file_path
        self._header = header
        self._first_record_index = first_record_index
        self._number_of_records = number_of_records
        self._samples_per_record = samples_per_record
        self._sample_offsets_in_record = sample_offsets_in_record

    def get_num_samples(self) -> int:
        return int(self._number_of_records * self._samples_per_record)

    def _resolve_offsets(self, channel_indices) -> np.ndarray:
        if channel_indices is None:
            channel_indices = slice(None)
        return np.atleast_1d(self._sample_offsets_in_record[channel_indices])

    def _resolve_positions(self, channel_indices) -> np.ndarray:
        """Channel positions in the recording's own order, needed to look up per-channel transforms."""
        if channel_indices is None:
            channel_indices = slice(None)
        return np.atleast_1d(np.arange(len(self._sample_offsets_in_record))[channel_indices])

    def get_traces(self, start_frame=None, end_frame=None, channel_indices=None) -> np.ndarray:
        number_of_samples = self.get_num_samples()
        start_frame = 0 if start_frame is None else max(int(start_frame), 0)
        end_frame = number_of_samples if end_frame is None else min(int(end_frame), number_of_samples)
        selected_offsets = self._resolve_offsets(channel_indices)
        selected_positions = self._resolve_positions(channel_indices)
        if end_frame <= start_frame:
            return np.empty((0, len(selected_offsets)), dtype=self._dtype)

        samples_per_record = self._samples_per_record
        first_record = start_frame // samples_per_record
        last_record = (end_frame - 1) // samples_per_record
        span_of_records = last_record - first_record + 1

        # Only the records overlapping the request are touched, in one contiguous seek + read.
        absolute_first_record = self._first_record_index + first_record
        byte_start = self._header.header_size_bytes + absolute_first_record * self._header.record_size_bytes
        with open(self._file_path, "rb") as file:
            file.seek(byte_start)
            raw = file.read(span_of_records * self._header.record_size_bytes)

        records = np.frombuffer(raw, dtype="<i2").reshape(span_of_records, self._header.samples_per_record_total)

        # Each signal occupies a contiguous block within every record, so a channel's samples over the
        # span are that block flattened in record order.
        traces = np.empty((span_of_records * samples_per_record, len(selected_offsets)), dtype=self._dtype)
        for column, (offset, position) in enumerate(zip(selected_offsets, selected_positions)):
            counts = records[:, offset : offset + samples_per_record].reshape(-1)
            transform = self._log_transforms.get(int(position))
            # A logarithmically transformed signal is decoded here rather than through the recording's
            # gain and offset, because the relationship between stored integer and physical value is
            # exponential and an ElectricalSeries applies only a scalar gain and offset.
            traces[:, column] = transform.decode(counts) if transform else counts

        trim_start = start_frame - first_record * samples_per_record
        return traces[trim_start : trim_start + (end_frame - start_frame)]


class EDFDRecordingExtractor(BaseRecording):
    """
    Lazy SpikeInterface recording for an EDF/EDF+ file, including discontinuous (``EDF+D``) files.

    Every contiguous run of data records becomes one segment, placed on the original session timeline
    through its ``t_start``. A file labelled ``EDF+D`` whose records are in fact contiguous yields a
    single segment — the common case, since exporters routinely label continuous recordings ``EDF+D``.

    ``EDF Annotations`` signals are never exposed as data channels; their contents are available as
    ``annotations_from_file`` and the run layout as ``runs``.

    Parameters
    ----------
    file_path : FilePath
        Path to the ``.edf`` file.
    channels_to_skip : list, optional
        Labels of channels to leave out of the recording. Applied while reading, so channels sampled at
        a rate differing from the rest can be dropped before the uniform-rate check rejects the file.
    """

    extractor_name = "EDFDRecording"
    mode = "file"
    name = "edfd"

    def __init__(self, file_path: FilePath, channels_to_skip: list | None = None):
        with open(file_path, "rb") as file:
            header = read_edf_header(file)
            if header.annotations_signal_indices:
                onsets, annotations = read_record_onsets_and_annotations(file=file, header=header)
                if not header.is_discontinuous:
                    # A continuous file's records are contiguous by definition; trust the geometry
                    # rather than the time-keeping TALs, which are optional for EDF+C.
                    onsets = np.arange(header.number_of_records, dtype="float64") * header.record_duration
            else:
                if header.is_discontinuous:
                    raise ValueError(
                        "This EDF+ file is marked discontinuous (EDF+D) but has no 'EDF Annotations' "
                        "signal, so the start time of each data record cannot be recovered. The file "
                        "does not conform to EDF+."
                    )
                onsets = np.arange(header.number_of_records, dtype="float64") * header.record_duration
                annotations = []

        layout = _build_channel_layout(header=header, channels_to_skip=channels_to_skip)
        sampling_frequency = layout["sampling_frequency"]

        runs = group_records_into_runs(
            onsets=onsets,
            samples_per_record=layout["samples_per_record"],
            sampling_frequency=sampling_frequency,
        )

        super().__init__(
            sampling_frequency=sampling_frequency,
            channel_ids=list(layout["channel_names"]),
            dtype=layout["dtype"],
        )
        for run in runs:
            self.add_recording_segment(
                EDFDRecordingSegment(
                    file_path=str(file_path),
                    header=header,
                    first_record_index=run.first_record_index,
                    number_of_records=run.number_of_records,
                    samples_per_record=layout["samples_per_record"],
                    sample_offsets_in_record=layout["sample_offsets_in_record"],
                    sampling_frequency=sampling_frequency,
                    t_start=run.start_time,
                    log_transforms=layout["log_transforms"],
                    dtype=layout["dtype"],
                )
            )

        # Gains/offsets take the stored digital values to microvolts; the ecephys writer then scales
        # microvolts to the volts that NWB's ElectricalSeries expects.
        self.set_channel_gains(gains=layout["gains_to_microvolts"])
        self.set_channel_offsets(offsets=layout["offsets_to_microvolts"])
        self.set_property(key="channel_name", values=np.asarray(layout["channel_names"]))
        # SpikeInterface treats this as a reserved property — it describes the ElectricalSeries rather
        # than becoming an electrodes column — and the neo-backed path sets it, so the two readers
        # would otherwise disagree about whether a caller can ask what unit the header declared.
        self.set_property(key="physical_unit", values=np.asarray(layout["physical_units"]))

        self.edf_header = header
        self.annotations_from_file = annotations
        self.runs = runs
        # Keyed by channel label, matching what the SpikeInterface path reports, so the interface's
        # log_transformed_channels means the same thing whichever reader produced the recording.
        self.log_transforms = {
            layout["channel_names"][position]: transform for position, transform in layout["log_transforms"].items()
        }
        self._kwargs = dict(file_path=str(file_path), channels_to_skip=channels_to_skip)
