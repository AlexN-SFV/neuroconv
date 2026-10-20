import warnings
from typing import Literal

from pydantic import FilePath
from pynwb import NWBFile

from ..baserecordingextractorinterface import BaseRecordingExtractorInterface
from ....utils import DeepDict


class BrainVisionRecordingInterface(BaseRecordingExtractorInterface):
    """
    Data interface class for converting BrainVision Core Data Format recordings (``.vhdr``/``.vmrk``/``.eeg``).

    The files are read by neuroconv's own reader rather than neo's, which misreads floating-point files
    whose resolution is not 1, cannot decode headers written in Latin-1, and places every marker one
    sample late. Samples are read lazily from the memory-mapped ``.eeg`` file.

    A recording that was paused and resumed carries one ``New Segment`` marker per resumption, and an
    epoched export from BrainVision Analyzer (``SegmentationType=MARKERBASED``) one per epoch. Each run
    of samples between them is written as its own ``ElectricalSeries``, sharing one electrodes table, as
    discontinuous EDF+ is. A run is placed at the time its marker's date gives when an earlier dated run
    gives that date something to be measured from, and directly after the run before it otherwise. Markers are written to a ``TimeIntervals``
    table named ``"markers"``. Several BrainVision recordings written into one NWB file share that table,
    and the ``"segments"`` table, with a ``source_file`` column saying which recording each row came from.
    """

    display_name = "BrainVision Recording"
    keywords = BaseRecordingExtractorInterface.keywords + ("BrainVision", "Brain Products", "EEG", "iEEG")
    associated_suffixes = (".vhdr", ".vmrk", ".eeg")
    info = "Interface for BrainVision Core Data Format recording data."

    @classmethod
    def get_source_schema(cls) -> dict:
        source_schema = super().get_source_schema()
        source_schema["properties"]["file_path"]["description"] = "Path to the .vhdr header file."
        return source_schema

    @staticmethod
    def get_available_channel_ids(file_path: FilePath) -> list:
        """
        Get the names of all channels in a BrainVision recording, from its header alone.

        Parameters
        ----------
        file_path : FilePath
            Path to the ``.vhdr`` header file.

        Returns
        -------
        list
            The channel names, in file order.
        """
        from ._brainvision_reader import read_brainvision_header

        return list(read_brainvision_header(file_path).channel_names)

    @classmethod
    def get_extractor_class(cls):
        from ._brainvision_extractor import BrainVisionRecordingExtractor

        return BrainVisionRecordingExtractor

    def _initialize_extractor(self, interface_kwargs: dict):
        self.extractor_kwargs = dict(
            file_path=interface_kwargs["file_path"],
            channels_to_skip=interface_kwargs.get("channels_to_skip"),
        )
        return self.get_extractor_class()(**self.extractor_kwargs)

    def __init__(
        self,
        file_path: FilePath,
        *,
        verbose: bool = False,
        es_key: str | None = None,
        metadata_key: str | None = None,
        channels_to_skip: list | None = None,
    ):
        """
        Load and prepare data for BrainVision.

        Parameters
        ----------
        file_path : str or Path
            Path to the ``.vhdr`` header file. The ``.eeg`` and ``.vmrk`` files it names are read from beside
            it, or, if they have been renamed without updating the header, by the header's own base name.
        verbose : bool, default: False
            Allows verbose.
        es_key : str, default: "ElectricalSeries"
            Deprecated. Use ``metadata_key`` instead.
        metadata_key : str, optional
            Key that indexes this interface's entries in the dict-based metadata. Defaults to
            ``"brainvision_recording"``.
        channels_to_skip : list, default: None
            Names of channels to leave out, such as auxiliary channels that are not neural data.
        """
        super().__init__(
            file_path=file_path,
            verbose=verbose,
            es_key=es_key,
            metadata_key=metadata_key,
            channels_to_skip=channels_to_skip,
        )
        if metadata_key is None:
            self.metadata_key = "brainvision_recording"

        self.brainvision_header = self.recording_extractor.brainvision_header

    @property
    def number_of_segments(self) -> int:
        """Number of uninterrupted runs of samples, 1 for a recording that was never paused."""
        return self._number_of_segments

    def get_metadata(self, *, use_new_metadata_format: bool = True) -> DeepDict:
        metadata = super().get_metadata(use_new_metadata_format=use_new_metadata_format)
        recording_start = self.brainvision_header.recording_start
        if recording_start is not None:
            metadata["NWBFile"]["session_start_time"] = recording_start
        return metadata

    def add_to_nwbfile(
        self,
        nwbfile: NWBFile,
        metadata: dict | None = None,
        *,
        stub_test: bool = False,
        write_as: Literal["raw", "lfp", "processed"] | None = None,
        parent_container: Literal["acquisition", "processing/LFP", "processing/FilteredEphys"] = "acquisition",
        data_representation: Literal["digital_counts", "physical_units"] = "digital_counts",
        write_electrical_series: bool = True,
        iterator_type: str | None = "v2",
        iterator_options: dict | None = None,
        always_write_timestamps: bool = False,
        write_markers: bool = True,
        write_segments: bool = True,
    ):
        """
        Add the BrainVision data to an NWBFile.

        Parameters
        ----------
        nwbfile : NWBFile
            The NWBFile to add the data to.
        metadata : dict, optional
            Metadata dictionary for constructing the NWBFile.
        stub_test : bool, default: False
            If True, truncate the data to run the conversion faster and take up less memory.
        write_as : {'raw', 'processed', 'lfp'}, optional
            Deprecated alias for ``parent_container``.
        parent_container : {'acquisition', 'processing/LFP', 'processing/FilteredEphys'}
            The NWB container the electrical series is written to.
        data_representation : {'digital_counts', 'physical_units'}, default='digital_counts'
            Whether to store the samples as read with the gain alongside, or as float physical values. See
            ``BaseRecordingExtractorInterface.add_to_nwbfile``.
        write_electrical_series : bool, default: True
            If False, only device, electrode groups and electrodes are written.
        iterator_type : {'v2', None}, default: 'v2'
            The type of iterator used for chunked data writing.
        iterator_options : dict, optional
            Options controlling the iterative write.
        always_write_timestamps : bool, default: False
            If True, always write an explicit timestamps array rather than a starting time and rate.
        write_markers : bool, default: True
            If True, the ``.vmrk`` markers other than ``New Segment`` are written to a ``TimeIntervals``
            table named ``"markers"``, at their times on the session timeline. If another BrainVision
            interface has already written that table to the file, the rows are added to it.
        write_segments : bool, default: True
            If True and the recording has more than one segment, the segment boundaries are written to a
            ``TimeIntervals`` table named ``"segments"``, or added to it as for the markers.
        """
        # The full signature is redeclared rather than taking **conversion_options, because
        # get_conversion_options_schema is built from it.
        conversion_options = dict(
            stub_test=stub_test,
            parent_container=parent_container,
            data_representation=data_representation,
            write_electrical_series=write_electrical_series,
            iterator_type=iterator_type,
            iterator_options=iterator_options,
            always_write_timestamps=always_write_timestamps,
        )
        if write_as is not None:
            conversion_options["write_as"] = write_as
            conversion_options.pop("parent_container")
        super().add_to_nwbfile(nwbfile=nwbfile, metadata=metadata, **conversion_options)

        if write_markers:
            self._add_markers_to_nwbfile(nwbfile=nwbfile)
        # A single segment is indistinguishable from an uninterrupted recording, so there is nothing to record.
        if write_segments and len(self.brainvision_header.segments) > 1:
            self._add_segments_to_nwbfile(nwbfile=nwbfile)

    def _time_of_sample(self, sample_index: int) -> float:
        """
        Session time of a sample, given as an index into the whole ``.eeg`` file.

        Read through the recording's own segment start times, so any alignment applied to the interface
        moves the markers with the data.
        """
        segments = self.brainvision_header.segments
        segment_index = 0
        for index, segment in enumerate(segments):
            if sample_index >= segment.first_sample:
                segment_index = index
        segment = segments[segment_index]
        start_time = float(self.recording_extractor.get_start_time(segment_index=segment_index))
        return start_time + (sample_index - segment.first_sample) / self.brainvision_header.sampling_frequency

    def _get_or_create_table(self, nwbfile: NWBFile, name: str, description: str, columns: dict):
        """
        Return the ``TimeIntervals`` table ``name`` to add rows to, creating it if the file has none.

        A table of that name already in the file is extended when it was written by another BrainVision
        interface (as when two recordings go into one file through a ``ConverterPipe``), so neither
        recording's rows are lost. A table of that name with other columns came from something else and is
        left alone, with a warning; None is returned.
        """
        from pynwb.epoch import TimeIntervals

        columns = dict(columns, source_file="Name of the BrainVision .vhdr header the row was read from.")
        if nwbfile.intervals is not None and name in nwbfile.intervals:
            table = nwbfile.intervals[name]
            expected = {"start_time", "stop_time", *columns}
            if set(table.colnames) != expected:
                warnings.warn(
                    f"The NWB file already has an intervals table named {name!r}, with columns "
                    f"{sorted(table.colnames)} rather than a BrainVision {name} table's {sorted(expected)}. "
                    f"The {name} of '{self.brainvision_header.header_path.name}' are not written.",
                    UserWarning,
                    stacklevel=3,
                )
                return None
            return table

        table = TimeIntervals(name=name, description=description)
        for column_name, column_description in columns.items():
            table.add_column(name=column_name, description=column_description)
        nwbfile.add_time_intervals(table)
        return table

    def _add_markers_to_nwbfile(self, nwbfile: NWBFile):
        header = self.brainvision_header
        markers = [marker for marker in header.markers if not marker.is_new_segment]
        # A marker positioned before the first sample or after the last one has no time in the recording.
        # It is left out, rather than moved to the nearest sample, which would misstate when it happened.
        outside = [marker for marker in markers if not 0 <= marker.sample_index < header.number_of_samples]
        if outside:
            shown = ", ".join(
                f"{marker.marker_type} {marker.description!r} at position {marker.sample_index + 1}"
                for marker in outside[:5]
            )
            warnings.warn(
                f"Skipping {len(outside)} marker(s) in '{header.marker_path.name}' positioned outside the "
                f"{header.number_of_samples} samples of '{header.data_path.name}' ({shown}).",
                UserWarning,
                stacklevel=3,
            )
            markers = [marker for marker in markers if 0 <= marker.sample_index < header.number_of_samples]
        if not markers:
            return

        table = self._get_or_create_table(
            nwbfile=nwbfile,
            name="markers",
            description="Markers imported from BrainVision .vmrk files.",
            columns={
                "marker_type": "The marker's type, such as 'Stimulus' or 'Comment'.",
                # "label" rather than "description", which TimeIntervals reserves for the table's own
                # description; the EDF+ annotations table uses the same column name.
                "label": "The marker's description, such as 'S  1'.",
                "size_in_samples": (
                    "The marker's size as the .vmrk file states it, in samples. A marker sits on a sample, so an "
                    "interval spans at least that one sample even where the file states a size of 0."
                ),
                "channel": "Name of the channel the marker refers to, or an empty string for all channels.",
            },
        )
        if table is None:
            return

        channel_names = header.channel_names
        source_file = header.header_path.name
        rows = []
        for marker in markers:
            start_time = self._time_of_sample(marker.sample_index)
            channel = ""
            if 1 <= marker.channel_number <= len(channel_names):
                channel = channel_names[marker.channel_number - 1]
            rows.append((start_time, marker, channel))
        # .vmrk files are numbered in the order markers were set, which need not be the order they occur in.
        rows.sort(key=lambda row: row[0])
        for start_time, marker, channel in rows:
            table.add_row(
                start_time=start_time,
                stop_time=start_time + max(marker.size, 1) / header.sampling_frequency,
                marker_type=marker.marker_type,
                label=marker.description,
                size_in_samples=marker.size,
                channel=channel,
                source_file=source_file,
            )

    def _add_segments_to_nwbfile(self, nwbfile: NWBFile):
        table = self._get_or_create_table(
            nwbfile=nwbfile,
            name="segments",
            description=(
                "Runs of samples in BrainVision recordings, each opened by a 'New Segment' marker: a resumption "
                "after a pause in a continuous recording, or one epoch of an epoched (SegmentationType=MARKERBASED) "
                "BrainVision Analyzer export. Each run is written as its own ElectricalSeries, positioned at the "
                "run's start time; epochs cut from overlapping windows overlap in time."
            ),
            columns={
                "segment_index": "0-based index of the run of samples within its recording.",
                "is_epoch": "True for an epoch of an epoched export, False for a run of a continuous recording.",
            },
        )
        if table is None:
            return

        header = self.brainvision_header
        for segment_index, segment in enumerate(header.segments):
            start_time = float(self.recording_extractor.get_start_time(segment_index=segment_index))
            table.add_row(
                start_time=start_time,
                stop_time=start_time + segment.number_of_samples / header.sampling_frequency,
                segment_index=segment_index,
                is_epoch=header.is_epoched,
                source_file=header.header_path.name,
            )
