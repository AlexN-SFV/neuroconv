import warnings
from typing import Literal, Optional

import numpy as np
from pydantic import FilePath
from pynwb import NWBFile

from .._channel_ids import _stacklevel_outside_neuroconv
from .._intervals_tables import add_rows_to_intervals_table
from ..baserecordingextractorinterface import BaseRecordingExtractorInterface
from ..basesortingextractorinterface import BaseSortingExtractorInterface
from ....utils import DeepDict, get_json_schema_from_method_signature

_NEV_EVENTS_TABLE = "nev_events"
_NEV_EVENTS_DESCRIPTION = (
    "Events from Blackrock .nev files: changes of the digital and serial input ports, analog input threshold "
    "crossings (FileSpec 2.1 and 2.2), and comments. Each spans one sample of the recording it was written with."
)
_NEV_EVENTS_COLUMNS = {
    "kind": (
        "What inserted the packet: 'digital_input', 'serial_input', 'analog_input', 'periodic_sampling', 'unknown', "
        "'comment', or 'tracking_event' (a NeuroMotive event stored as a comment)."
    ),
    # "label" rather than "description", which TimeIntervals reserves for the table's own description.
    "label": (
        "The digital or serial input word in decimal, the analog input values, or the comment's text up to its "
        "terminator."
    ),
    "value": "The digital or serial input word, -1 for a comment.",
    "packet_insertion_reason": "The packet's insertion reason byte as the .nev stores it, -1 for a comment.",
}
_NEV_SOURCE_FILE_DESCRIPTION = "Name of the Blackrock .nev file the row was read from."

_RUNS_TABLE = "runs"
_RUNS_DESCRIPTION = (
    "Uninterrupted runs of samples in Blackrock recordings paused, or with their clock reset, during recording. "
    "Each run is written as its own ElectricalSeries, positioned at the run's start time."
)
_RUNS_COLUMNS = {
    "run_index": "0-based index of the uninterrupted run of samples within its recording.",
    "after_clock_reset": (
        "Whether the run follows a reset of the recording's clock. Its true start is not in the file; it is placed "
        "where the run before it ends plus its own clock value since the reset, the earliest it can have started."
    ),
}
_RUNS_SOURCE_FILE_DESCRIPTION = "Name of the Blackrock .nsX file the row was read from."


def _warn_if_nev_time_may_be_local(nev, use: str) -> None:
    if nev.time_origin_may_be_local():
        warnings.warn(
            f"The session start time is taken from the .nev file '{nev.path.name}' ({use}), written by "
            f"'{nev.creating_application}'. The format defines it as UTC and it is read as UTC, but File Dialog "
            "versions before 6.05 have been seen writing local time there instead (FileSpec2.3001.nev, v6.04, is "
            "seven hours behind its own .ns5). Set metadata['NWBFile']['session_start_time'] where the time zone "
            "is known.",
            UserWarning,
            stacklevel=_stacklevel_outside_neuroconv(),
        )


class BlackrockRecordingInterface(BaseRecordingExtractorInterface):
    """
    Primary data interface class for converting Blackrock continuous data (``.ns1`` to ``.ns6``).

    neo's ``BlackrockRawIO`` parses the headers and memory-maps the samples, and neuroconv places them, so that
    every file specification lands on one consistent timeline:

    * ``session_start_time`` is the ``.nsX`` header's ``TimeOrigin``, which is UTC, and the earliest first
      sample among the set's ``.nsX`` files is at 0, so the streams of one set, and their ``.nev``, line up in one
      NWB file. (FileSpec 3.0 clocks do not start with the recording, and PTP clocks count Unix-epoch
      nanoseconds.)
    * A recording paused is written as one ``ElectricalSeries`` per run, each at its start time. A run after
      a clock reset, whose real start the file does not hold, is placed where the run before it ends.
    * PTP (Gemini) packets are read in timestamp order, and a run ends where a sample is missing.
    * FileSpec 2.1 samples are scaled with the ``.nev`` file's digitization factors.

    The ``.nev`` file recorded with the stream, when there is one, provides its digital and serial input
    events and its comments, written to a ``TimeIntervals`` table named ``"nev_events"``; the runs, when there
    is more than one, go to one named ``"runs"``. Several recordings written into one NWB file share those
    tables, with a ``source_file`` column. Spikes are the job of :py:class:`BlackrockSortingInterface`.
    """

    display_name = "Blackrock Recording"
    keywords = BaseRecordingExtractorInterface.keywords + ("Blackrock", "Cerebus", "Neuroport", "NSx", "NEV")
    associated_suffixes = (".ns1", ".ns2", ".ns3", ".ns4", ".ns5", ".ns6", ".nev")
    info = "Interface for Blackrock recording data."

    @classmethod
    def get_source_schema(cls):
        source_schema = get_json_schema_from_method_signature(method=cls.__init__, exclude=["block_index", "seg_index"])
        source_schema["properties"]["file_path"][
            "description"
        ] = "Path to the Blackrock continuous data file, with suffix .ns1, .ns2, .ns3, .ns4, .ns5 or .ns6."
        return source_schema

    @classmethod
    def get_extractor_class(cls):
        from ._blackrock_extractor import BlackrockRecordingExtractor

        return BlackrockRecordingExtractor

    def _initialize_extractor(self, interface_kwargs: dict):
        self.extractor_kwargs = dict(
            file_path=interface_kwargs["file_path"],
            nsx_override=interface_kwargs.get("nsx_override"),
            nev_file_path=interface_kwargs.get("nev_file_path"),
        )
        return self.get_extractor_class()(**self.extractor_kwargs)

    def __init__(
        self,
        file_path: FilePath,
        *,
        nsx_override: FilePath | None = None,
        nev_file_path: FilePath | None = None,
        verbose: bool = False,
        es_key: str | None = None,
        metadata_key: str | None = None,
    ):
        """
        Load and prepare data corresponding to Blackrock interface.

        Parameters
        ----------
        file_path : FilePath
            Path to the Blackrock continuous data file, with suffix .ns1, .ns2, .ns3, .ns4, .ns5 or .ns6. Base names
            with dots of their own, such as ``FileSpec2.3001.ns5``, are read as they are. A base name without
            suffix is accepted too, when exactly one .nsX file has it.
        nsx_override : FilePath, optional
            Path to the .nsX file, when ``file_path`` names the file set instead (its .nev file or base name) and
            the .nsX file has another base name.
        nev_file_path : FilePath, optional
            Path to the .nev file recorded with the stream, when it does not share the .nsX file's base name. It
            provides the events, and, for FileSpec 2.1, the scaling and start time.
        verbose : bool, default: False
        es_key : str, optional
            Deprecated. Use ``metadata_key`` instead.
        metadata_key : str, optional
            Key that indexes this interface's entries in the dict-based metadata. Defaults to
            ``"blackrock_recording"``.
        """
        super().__init__(
            file_path=file_path,
            nsx_override=nsx_override,
            nev_file_path=nev_file_path,
            verbose=verbose,
            es_key=es_key,
            metadata_key=metadata_key,
        )
        if metadata_key is None:
            self.metadata_key = "blackrock_recording"

        self.blackrock_stream = self.recording_extractor.blackrock_stream
        self.blackrock_nev = self.recording_extractor.blackrock_nev
        self.file_path = self.blackrock_stream.path

    def get_metadata(self, *, use_new_metadata_format: bool = True) -> DeepDict:
        metadata = super().get_metadata(use_new_metadata_format=use_new_metadata_format)
        stream = self.blackrock_stream
        if stream.time_origin is not None:
            metadata["NWBFile"]["session_start_time"] = stream.time_origin
        elif self.blackrock_nev is not None:
            # FileSpec 2.1 .nsX headers have no time; the .nev recorded with them does.
            _warn_if_nev_time_may_be_local(self.blackrock_nev, use=f"'{stream.path.name}' has no time of its own")
            metadata["NWBFile"]["session_start_time"] = self.blackrock_nev.time_origin
        if stream.comment:
            metadata["NWBFile"]["session_description"] = stream.comment
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
        write_nev_events: bool = True,
        write_runs: bool = True,
    ):
        """
        Add the Blackrock data to an NWBFile.

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
            Whether to store the counts as read with the gain alongside, or as float physical values. See
            ``BaseRecordingExtractorInterface.add_to_nwbfile``.
        write_electrical_series : bool, default: True
            If False, only device, electrode groups and electrodes are written.
        iterator_type : {'v2', None}, default: 'v2'
            The type of iterator used for chunked data writing.
        iterator_options : dict, optional
            Options controlling the iterative write.
        always_write_timestamps : bool, default: False
            If True, always write an explicit timestamps array rather than a starting time and rate.
        write_nev_events : bool, default: True
            If True and the .nev file recorded with the stream holds events or comments, they are written, in time
            order, to a ``TimeIntervals`` table named ``"nev_events"``. If another Blackrock interface has already
            written that table to the file, the rows are added to it.
        write_runs : bool, default: True
            If True and the recording has more than one run, the runs are written to a ``TimeIntervals`` table
            named ``"runs"``, or added to it as for the events.
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

        if write_nev_events and self.blackrock_nev is not None and len(self.blackrock_nev.events.ticks):
            self._add_nev_events_to_nwbfile(nwbfile=nwbfile)
        # A single run is indistinguishable from an uninterrupted recording, so there is nothing to record.
        if write_runs and len(self.blackrock_stream.runs) > 1:
            self._add_runs_to_nwbfile(nwbfile=nwbfile)

    def _aligned(self, seconds: np.ndarray) -> np.ndarray:
        """
        Times the file places, moved with the run each falls in (or the run before it), so the tables follow any
        alignment of the series, segment by segment.
        """
        runs = self.blackrock_stream.runs
        file_starts = np.array([run.start_time for run in runs])
        shifts = np.array(
            [
                float(self.recording_extractor.get_start_time(segment_index=index)) - run.start_time
                for index, run in enumerate(runs)
            ]
        )
        # A time before the first run, or one left unplaced (NaN), takes the first run's shift.
        indices = np.clip(np.searchsorted(file_starts, np.nan_to_num(seconds, nan=-np.inf), side="right") - 1, 0, None)
        return seconds + shifts[indices]

    def _add_nev_events_to_nwbfile(self, nwbfile: NWBFile):
        nev, stream = self.blackrock_nev, self.blackrock_stream
        events = nev.events
        seconds = self._aligned(stream.timeline.to_seconds(events.ticks, nev.resolution, events.epochs))
        unplaced = ~np.isfinite(seconds)
        if unplaced.any():
            warnings.warn(
                f"{int(unplaced.sum())} events of '{nev.path.name}' fall after a clock reset that '{stream.path.name}' "
                "does not have, so their time is unknown; they are not written.",
                UserWarning,
                stacklevel=_stacklevel_outside_neuroconv(),
            )
        sample_period = 1.0 / stream.sampling_frequency
        rows = [
            dict(
                start_time=float(seconds[index]),
                # An event marks an instant; it spans one sample, because NWB Inspector flags an interval that
                # stops where it starts.
                stop_time=float(seconds[index]) + sample_period,
                kind=events.kinds[index],
                label=events.labels[index],
                value=int(events.values[index]),
                packet_insertion_reason=int(events.packet_insertion_reasons[index]),
            )
            for index in np.argsort(seconds, kind="stable")
            if not unplaced[index]
        ]
        add_rows_to_intervals_table(
            nwbfile=nwbfile,
            file_path=nev.path,
            name=_NEV_EVENTS_TABLE,
            description=_NEV_EVENTS_DESCRIPTION,
            columns=_NEV_EVENTS_COLUMNS,
            rows=rows,
            kind="Blackrock",
            source_file_description=_NEV_SOURCE_FILE_DESCRIPTION,
        )

    def _add_runs_to_nwbfile(self, nwbfile: NWBFile):
        stream = self.blackrock_stream
        rows = []
        for run_index, run in enumerate(stream.runs):
            start_time = float(self.recording_extractor.get_start_time(segment_index=run_index))
            rows.append(
                dict(
                    start_time=start_time,
                    stop_time=start_time + run.number_of_samples / stream.sampling_frequency,
                    run_index=run_index,
                    after_clock_reset=bool(run.after_clock_reset),
                )
            )
        add_rows_to_intervals_table(
            nwbfile=nwbfile,
            file_path=stream.path,
            name=_RUNS_TABLE,
            description=_RUNS_DESCRIPTION,
            columns=_RUNS_COLUMNS,
            rows=rows,
            kind="Blackrock",
            source_file_description=_RUNS_SOURCE_FILE_DESCRIPTION,
        )


class BlackrockSortingInterface(BaseSortingExtractorInterface):
    """
    Primary data interface class for converting Blackrock spiking data (``.nev``).

    Spike times are placed on the timeline :py:class:`BlackrockRecordingInterface` gives the ``.nsX`` stream
    recorded with them, so the two line up in one NWB file: the set's earliest first sample at 0, runs after a
    clock reset moved
    past the run before, PTP spikes on the samples they were recorded with. ``session_start_time`` is that
    stream's ``TimeOrigin``, which is UTC.
    """

    display_name = "Blackrock Sorting"
    associated_suffixes = (".nev",)
    info = "Interface for Blackrock sorting data."

    @classmethod
    def get_source_schema(cls) -> dict:
        metadata_schema = get_json_schema_from_method_signature(method=cls.__init__)
        metadata_schema["additionalProperties"] = True
        metadata_schema["properties"]["file_path"].update(description="Path to Blackrock .nev file.")
        return metadata_schema

    @classmethod
    def get_extractor_class(cls):
        from ._blackrock_extractor import BlackrockSortingExtractor

        return BlackrockSortingExtractor

    def __init__(
        self,
        file_path: FilePath,
        *,
        sampling_frequency: Optional[float] = None,
        nsx_to_load: Optional[int | list | str] = None,
        verbose: bool = False,
    ):
        """
        Parameters
        ----------
        file_path : str, Path
            The file path to the ``.nev`` data
        sampling_frequency : float, optional
            The rate spike frames are counted at. Defaults to the .nev header's sample resolution, 30 kHz.
        nsx_to_load : int | list | str, optional
            The .nsX streams of the same base name whose timeline may place the spikes: an int such as 5 for the
            .ns5 alone, a list of them, 'max' for the highest-numbered, or 'all' or None for every one. Every
            stream of a set is placed from the set's earliest first sample, so any of them gives the same spike
            times; of several, the smallest file is read. Pass the stream converted with
            :py:class:`BlackrockRecordingInterface` to anchor PTP spikes to its own samples. An empty list uses
            none: the spikes keep the .nev's clock, its first packet at 0 for FileSpec 3.0.
        verbose : bool, default: False
            Enables verbosity
        """

        super().__init__(
            file_path=file_path,
            sampling_frequency=sampling_frequency,
            nsx_to_load=nsx_to_load,
            verbose=verbose,
        )
        self.blackrock_nev = self.sorting_extractor.blackrock_nev
        self.blackrock_stream = self.sorting_extractor.blackrock_stream

    def get_metadata(self) -> DeepDict:
        metadata = super().get_metadata()
        stream, nev = self.blackrock_stream, self.blackrock_nev
        if stream is not None and stream.time_origin is not None:
            metadata["NWBFile"]["session_start_time"] = stream.time_origin
        else:
            _warn_if_nev_time_may_be_local(nev, use="no .nsX header gives one")
            metadata["NWBFile"]["session_start_time"] = nev.time_origin
        if stream is not None and stream.comment:
            metadata["NWBFile"]["session_description"] = stream.comment
        return metadata
