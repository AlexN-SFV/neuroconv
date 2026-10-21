"""
SpikeInterface recording extractor for BrainVision recordings.

The parsing lives in :mod:`._brainvision_reader`, which depends only on NumPy. This module is where
SpikeInterface enters, so it must only be imported lazily, as
:meth:`~..brainvisiondatainterface.BrainVisionRecordingInterface.get_extractor_class` does.

The classes are defined at module scope because SpikeInterface records an extractor's class by its
import path in ``to_dict()``; classes built inside a function cannot be pickled or reloaded.
"""

import numpy as np
from pydantic import FilePath
from spikeinterface.core import BaseRecording, BaseRecordingSegment

from ._brainvision_reader import BrainVisionHeader, memmap_samples, read_brainvision_header
from .._channel_ids import (
    indices_to_keep,
    set_original_channel_names,
    unknown_channels_to_skip,
    warn_about_renamed_channels,
)


class BrainVisionRecordingSegment(BaseRecordingSegment):
    """One uninterrupted run of samples, read lazily from the memory-mapped ``.eeg`` file."""

    def __init__(
        self,
        *,
        header: BrainVisionHeader,
        first_sample: int,
        number_of_samples: int,
        channel_indices: np.ndarray,
        t_start: float,
    ):
        super().__init__(sampling_frequency=header.sampling_frequency, t_start=t_start)
        self._header = header
        self._first_sample = first_sample
        self._number_of_samples = number_of_samples
        self._channel_indices = channel_indices

    def get_num_samples(self) -> int:
        return int(self._number_of_samples)

    def get_traces(self, start_frame=None, end_frame=None, channel_indices=None) -> np.ndarray:
        start_frame = 0 if start_frame is None else max(int(start_frame), 0)
        end_frame = self._number_of_samples if end_frame is None else min(int(end_frame), self._number_of_samples)
        columns = self._channel_indices if channel_indices is None else self._channel_indices[channel_indices]
        columns = np.atleast_1d(columns)
        if end_frame <= start_frame:
            return np.empty((0, len(columns)), dtype=self._header.dtype)

        # Mapped per call rather than held open, so the segment pickles and no file handle outlives a read.
        samples = memmap_samples(self._header)
        rows = slice(self._first_sample + start_frame, self._first_sample + end_frame)
        traces = samples[rows][:, columns]
        return np.ascontiguousarray(traces, dtype=np.dtype(self._header.dtype).newbyteorder("="))


class BrainVisionRecordingExtractor(BaseRecording):
    """
    Lazy SpikeInterface recording for a BrainVision recording.

    Each run of samples between ``New Segment`` markers becomes one segment, placed on the session
    timeline at the time its marker states. A recording that was never paused is a single segment.

    Parameters
    ----------
    file_path : FilePath
        Path to the ``.vhdr`` header file.
    channels_to_skip : list, optional
        Channels to leave out of the recording, by id or by name. A name the header gives more than one
        channel leaves out all of them; the id a repeat was renamed to leaves out that channel alone.
    """

    extractor_name = "BrainVisionRecording"
    mode = "file"
    name = "brainvision"

    def __init__(self, file_path: FilePath, channels_to_skip: list | None = None):
        header = read_brainvision_header(file_path)

        labels = header.channel_names
        ids = header.channel_ids
        unknown = unknown_channels_to_skip(labels=labels, ids=ids, channels_to_skip=channels_to_skip)
        if unknown:
            raise ValueError(
                f"channels_to_skip names channels that are not in the recording: {unknown}. "
                f"Available channels: {ids}."
            )
        channel_indices = np.array(indices_to_keep(labels=labels, ids=ids, channels_to_skip=channels_to_skip))
        if len(channel_indices) == 0:
            raise ValueError("channels_to_skip leaves no channels in the recording.")
        channel_names = [ids[index] for index in channel_indices]
        kept_labels = [labels[index] for index in channel_indices]
        warn_about_renamed_channels(
            labels=kept_labels, ids=channel_names, source=f"The BrainVision header '{header.header_path.name}'"
        )

        super().__init__(
            sampling_frequency=header.sampling_frequency,
            channel_ids=channel_names,
            dtype=np.dtype(header.dtype).newbyteorder("="),
        )
        for segment in header.segments:
            self.add_recording_segment(
                BrainVisionRecordingSegment(
                    header=header,
                    first_sample=segment.first_sample,
                    number_of_samples=segment.number_of_samples,
                    channel_indices=channel_indices,
                    t_start=segment.start_time,
                )
            )

        # Gains take the stored samples to microvolts; the ecephys writer then scales microvolts to the
        # volts NWB's ElectricalSeries expects. The format has no offset.
        self.set_channel_gains(gains=header.gains_to_microvolts[channel_indices])
        self.set_channel_offsets(offsets=np.zeros(len(channel_indices)))
        self.set_property(key="channel_name", values=np.asarray(channel_names))
        set_original_channel_names(recording=self, labels=kept_labels, ids=channel_names)
        # The unit the header declares, and the scaling into it rather than into microvolts, so a caller can
        # check the header against an external source such as a BIDS channels.tsv, as on the EDF readers.
        self.set_property(key="physical_unit", values=np.asarray([header.units[index] for index in channel_indices]))
        self.set_property(key="gain_to_physical_unit", values=header.resolutions[channel_indices])
        self.set_property(key="offset_to_physical_unit", values=np.zeros(len(channel_indices)))
        references = [header.references[index] for index in channel_indices]
        if any(references):
            self.set_property(key="reference", values=np.asarray(references))

        self.brainvision_header = header
        self._kwargs = dict(file_path=str(file_path), channels_to_skip=channels_to_skip)
