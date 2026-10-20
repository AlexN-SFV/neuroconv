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

    A recording that was paused and resumed carries one ``New Segment`` marker per resumption. Each run
    of samples between them is written as its own ``ElectricalSeries`` at the time its marker states,
    sharing one electrodes table, as discontinuous EDF+ is. Markers are written to a ``TimeIntervals``
    table named ``"markers"``.
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
            table named ``"markers"``, at their times on the session timeline.
        write_segments : bool, default: True
            If True and the recording has more than one segment, the segment boundaries are written to a
            ``TimeIntervals`` table named ``"segments"``.
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

    def _add_markers_to_nwbfile(self, nwbfile: NWBFile):
        from pynwb.epoch import TimeIntervals

        markers = [marker for marker in self.brainvision_header.markers if marker.marker_type != "New Segment"]
        if not markers:
            return
        # split_by_offset can produce several interfaces backed by the same files, written into one NWBFile.
        if nwbfile.intervals is not None and "markers" in nwbfile.intervals:
            return

        channel_names = self.brainvision_header.channel_names
        sampling_frequency = self.brainvision_header.sampling_frequency
        table = TimeIntervals(name="markers", description="Markers imported from the BrainVision .vmrk file.")
        table.add_column(name="marker_type", description="The marker's type, such as 'Stimulus' or 'Comment'.")
        # "label" rather than "description", which TimeIntervals reserves for the table's own description; the
        # EDF+ annotations table uses the same column name.
        table.add_column(name="label", description="The marker's description, such as 'S  1'.")
        table.add_column(
            name="size_in_samples",
            description=(
                "The marker's size as the .vmrk file states it, in samples. A marker sits on a sample, so an "
                "interval spans at least that one sample even where the file states a size of 0."
            ),
        )
        table.add_column(
            name="channel",
            description="Name of the channel the marker refers to, or an empty string for all channels.",
        )
        for marker in markers:
            start_time = self._time_of_sample(marker.sample_index)
            channel = ""
            if 1 <= marker.channel_number <= len(channel_names):
                channel = channel_names[marker.channel_number - 1]
            table.add_row(
                start_time=start_time,
                stop_time=start_time + max(marker.size, 1) / sampling_frequency,
                marker_type=marker.marker_type,
                label=marker.description,
                size_in_samples=marker.size,
                channel=channel,
            )
        nwbfile.add_time_intervals(table)

    def _add_segments_to_nwbfile(self, nwbfile: NWBFile):
        from pynwb.epoch import TimeIntervals

        if nwbfile.intervals is not None and "segments" in nwbfile.intervals:
            return

        sampling_frequency = self.brainvision_header.sampling_frequency
        table = TimeIntervals(
            name="segments",
            description=(
                "Uninterrupted runs of samples in the BrainVision recording, delimited by its 'New Segment' "
                "markers. Each run is written as its own ElectricalSeries, positioned at the run's start time."
            ),
        )
        table.add_column(name="segment_index", description="0-based index of the run of samples.")
        for segment_index, segment in enumerate(self.brainvision_header.segments):
            start_time = float(self.recording_extractor.get_start_time(segment_index=segment_index))
            table.add_row(
                start_time=start_time,
                stop_time=start_time + segment.number_of_samples / sampling_frequency,
                segment_index=segment_index,
            )
        nwbfile.add_time_intervals(table)
