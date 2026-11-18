import copy
import math
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import DirectoryPath, FilePath
from pynwb import NWBFile

from ._runs_table import add_runs_to_nwbfile
from .._channel_ids import _stacklevel_outside_neuroconv
from .._intervals_tables import add_rows_to_intervals_table
from ..baserecordingextractorinterface import BaseRecordingExtractorInterface
from ....utils import DeepDict

_RUNS_SOURCE_FILE_DESCRIPTION = (
    "Name of the MEF recording (.mefd session or directory of .mef files) the row was read from."
)
_ANNOTATIONS_SOURCE_FILE_DESCRIPTION = "Name of the annotations file the row was read from."

# No real MEF recording predates this; MEF 3 writers move a de-identified session to 1970-01-01.
_DEIDENTIFIED_BEFORE = datetime(1980, 1, 1, tzinfo=timezone.utc)
# The day the de-identified MEF 2 recordings of IEEG.org, as republished on Pennsieve Discover, start on.
_DEIDENTIFIED_DAY = datetime(2000, 1, 1, tzinfo=timezone.utc).date()


def _independent_selection(recording, positions: list[int]):
    """
    ``recording.select_segments(positions)``, with copies of the segments rather than the source's own.

    SpikeInterface's selection shares, and re-parents, the source's segment objects, and alignment sets the time
    of a segment object in place, so aligning one selection would move the source's runs and every other
    selection's. Each copy shares the decoding state, which is only read.
    """
    originals = [recording.segments[position] for position in positions]
    selection = recording.select_segments(positions)
    for segment in originals:
        segment.set_parent_extractor(recording)
    selection._segments = []
    for segment in originals:
        selection.add_recording_segment(copy.copy(segment))
    return selection


class MEF2RecordingInterface(BaseRecordingExtractorInterface):
    """
    Data interface class for converting MEF 2.0 and 2.1 recordings (Multiscale Electrophysiology Format, ``.mef``).

    A MEF 2 recording is a directory of ``.mef`` files, one per channel. It is read by neuroconv's own reader,
    with the RED decoder the MEF 3 interface uses; no MEF 2 library or MNE is needed. The samples are decoded
    lazily, block by block, and stored as the integer counts the files hold, with each channel's voltage
    conversion factor as its gain, sign included: a negative factor marks inverted counts.

    A recording with interruptions is written as one ``ElectricalSeries`` per uninterrupted run of blocks,
    sharing one electrodes table, each placed at its first block's start time at the nominal sampling rate.
    Runs break where the files flag a block as following a discontinuity, as for MEF 3. MEF 2 writers also
    set their clock back now and then while the samples run on; such a step, unflagged, stays inside the run.

    The runs, when there is more than one, are written to a ``TimeIntervals`` table named ``"runs"``, and the
    annotations of an IEEG.org ``annotations.xml`` to one named ``"annotations"``. Channels sampled at
    different rates are read one rate, or stream, at a time. Encrypted files are refused.

    A long recording can be written to several NWB files, each holding some of its runs: give ``run_indices``,
    or call :meth:`select_runs` on an interface over the whole recording, which reads the headers only once.
    :meth:`get_runs` lists the runs from the headers alone. Each run keeps the series name, starting time and
    samples it would have in a file of the whole recording, every file has the recording's
    ``session_start_time``, and the ``runs`` table always lists the file's runs by their index in the
    recording, so the files line up on one timeline.
    """

    display_name = "MEF 2 Recording"
    keywords = BaseRecordingExtractorInterface.keywords + (
        "MEF",
        "MEF2",
        "Multiscale Electrophysiology Format",
        "iEEG",
        "EEG",
        "IEEG.org",
    )
    associated_suffixes = (".mef",)
    info = "Interface for MEF 2.0 and 2.1 (Multiscale Electrophysiology Format) recording data."

    @classmethod
    def get_source_schema(cls) -> dict:
        source_schema = super().get_source_schema()
        source_schema["properties"]["folder_path"][
            "description"
        ] = "Path to the directory of the recording's .mef files."
        return source_schema

    @staticmethod
    def get_available_channel_ids(folder_path: DirectoryPath) -> list:
        """
        Get the ids of the channels of a MEF 2 recording, of every stream, reading only each file's header.

        Parameters
        ----------
        folder_path : DirectoryPath
            Path to the directory of the recording's ``.mef`` files.

        Returns
        -------
        list
            The channel ids, in physical channel order. A name two files share at one sampling rate is
            renamed ``<name>_2``, ``<name>_3``, ...
        """
        from ._mef2_reader import available_channel_names

        return available_channel_names(folder_path)

    @classmethod
    def get_stream_names(cls, folder_path: DirectoryPath) -> list[str]:
        """
        Get the streams of a MEF 2 recording: one per sampling rate, named like ``"500 Hz"``, slowest first.

        Parameters
        ----------
        folder_path : DirectoryPath
            Path to the directory of the recording's ``.mef`` files.

        Returns
        -------
        list of str
        """
        from ._mef2_reader import stream_names

        return stream_names(folder_path)

    @staticmethod
    def get_runs(
        folder_path: DirectoryPath, *, stream_name: str | None = None, channels_to_skip: list | None = None
    ) -> list:
        """
        List the uninterrupted runs of a MEF 2 recording from its headers and block indices, decoding no samples.

        Parameters
        ----------
        folder_path : DirectoryPath
            Path to the directory of the recording's ``.mef`` files.
        stream_name : str, optional
            The sampling rate, as ``get_stream_names`` names it; required when the files have more than one.
        channels_to_skip : list, optional
            Channels to leave out of the byte counts, as for the interface.

        Returns
        -------
        list of MEF2RunSummary
            One per run, in time order: ``index``, ``start_offset`` (seconds from the recording's start, where
            the run's series starts), ``number_of_samples`` and ``number_of_bytes`` (the compressed blocks of
            every kept channel, a measure of the work and output the run makes).
        """
        from ._mef2_reader import read_mef2_recording, run_summaries

        recording = read_mef2_recording(folder_path, channels_to_skip=channels_to_skip, stream_name=stream_name)
        return run_summaries(recording)

    @classmethod
    def get_extractor_class(cls):
        from ._mef2_extractor import MEF2RecordingExtractor

        return MEF2RecordingExtractor

    def _initialize_extractor(self, interface_kwargs: dict):
        self.extractor_kwargs = dict(
            folder_path=interface_kwargs["folder_path"],
            channels_to_skip=interface_kwargs.get("channels_to_skip"),
            stream_name=interface_kwargs.get("stream_name"),
            validate_crc=interface_kwargs.get("validate_crc", False),
            run_indices=interface_kwargs.get("run_indices"),
        )
        return self.get_extractor_class()(**self.extractor_kwargs)

    def __init__(
        self,
        folder_path: DirectoryPath,
        *,
        verbose: bool = False,
        metadata_key: str | None = None,
        channels_to_skip: list | None = None,
        stream_name: str | None = None,
        annotations_file_path: FilePath | None = None,
        validate_crc: bool = False,
        run_indices: list[int] | None = None,
    ):
        """
        Load and prepare data for MEF 2.

        Parameters
        ----------
        folder_path : str or Path
            Path to the directory of the recording's ``.mef`` files.
        verbose : bool, default: False
            Allows verbose.
        metadata_key : str, optional
            Key that indexes this interface's entries in the dict-based metadata. Defaults to
            ``"mef2_recording"``.
        channels_to_skip : list, default: None
            Channels to leave out, by id or by the name the file gives them, such as channels that are not
            neural data.
        stream_name : str, optional
            The sampling rate to convert, as ``get_stream_names`` names it, such as ``"500 Hz"``. Required when
            the files are sampled at more than one rate; convert each stream with its own interface. To write
            several streams to one NWB file, give each interface its own ``metadata_key`` and its series its
            own name under that key, as the MEF 2 gallery page shows.
        annotations_file_path : str or Path, optional
            An IEEG.org ``annotations.xml`` for the recording. Defaults to ``annotations.xml`` in
            ``folder_path`` when there is one.
        validate_crc : bool, default: False
            If True, check each block's CRC as it is decoded and raise on a block or MEF 2.1 header that fails
            it, at a cost of a few percent of decoding time. Otherwise a failing header CRC only warns and block
            CRCs are not computed.
        run_indices : list of int, optional
            Write only these runs, by their 0-based index in the whole recording (see ``get_runs``), contiguous or
            not. Each keeps the series name and starting time it has in a file of the whole recording, the runs
            table lists them, even a single one, by that index, and only the annotations that overlap them are
            written, each run taking the gap after it. Defaults to every run.
        """
        super().__init__(
            folder_path=folder_path,
            verbose=verbose,
            metadata_key=metadata_key,
            channels_to_skip=channels_to_skip,
            stream_name=stream_name,
            validate_crc=validate_crc,
            run_indices=run_indices,
        )
        if metadata_key is None:
            self.metadata_key = "mef2_recording"

        self.mef2_recording = self.recording_extractor.mef2_recording
        # The index in the whole recording of each segment's run, and whether a selection was asked for.
        self.run_indices = list(self.recording_extractor.run_indices)
        self.runs_selected = run_indices is not None
        if annotations_file_path is None:
            default = Path(folder_path) / "annotations.xml"
            annotations_file_path = default if default.is_file() else None
        self.annotations_file_path = None if annotations_file_path is None else Path(annotations_file_path)
        self._annotations = None
        if self.annotations_file_path is not None:
            from ._mef2_reader import read_ieeg_annotations

            self._annotations = read_ieeg_annotations(
                self.annotations_file_path, recording_start_time=self.mef2_recording.recording_start_time
            )

    def describe_runs(self) -> list:
        """
        The runs of the whole recording, as ``get_runs`` lists them, from the headers this interface has read.

        Returns
        -------
        list of MEF2RunSummary
        """
        from ._mef2_reader import run_summaries

        return run_summaries(self.mef2_recording)

    def select_runs(self, run_indices: list[int]) -> "MEF2RecordingInterface":
        """
        An interface over some of this interface's runs, sharing its headers, channel properties and annotations.

        For writing a long recording to several files without reading its headers again for each. Channel
        properties set on this interface's extractor, and its aligned starting times, carry over. The selection
        has segments of its own: aligning it later leaves this interface's runs where they are.

        Parameters
        ----------
        run_indices : list of int
            Runs by their 0-based index in the whole recording; each must be one of this interface's runs.

        Returns
        -------
        MEF2RecordingInterface
        """
        from ._mef2_extractor import check_run_indices

        indices = check_run_indices(
            run_indices, number_of_runs=len(self.mef2_recording.runs), name=self.mef2_recording.path.name
        )
        positions = {run_index: position for position, run_index in enumerate(self.run_indices)}
        missing = [index for index in indices if index not in positions]
        if missing:
            raise ValueError(f"Runs {missing[:5]} are not among this interface's runs.")
        selected = copy.copy(self)
        selected.recording_extractor = _independent_selection(self.recording_extractor, [positions[i] for i in indices])
        selected.run_indices = indices
        selected._number_of_segments = len(indices)
        selected.runs_selected = True
        selected.source_data = dict(self.source_data, run_indices=indices)
        return selected

    def get_metadata(self, *, use_new_metadata_format: bool = True) -> DeepDict:
        metadata = super().get_metadata(use_new_metadata_format=use_new_metadata_format)
        session_start_time = self.mef2_recording.start_datetime
        utc_start = session_start_time.astimezone(timezone.utc)
        if utc_start < _DEIDENTIFIED_BEFORE or utc_start.date() == _DEIDENTIFIED_DAY:
            warnings.warn(
                f"The MEF 2 recording '{self.mef2_recording.path.name}' starts at {session_start_time.isoformat()}, "
                "on the date de-identified recordings are moved to, not when it was recorded. It is used as "
                "session_start_time as it is; set metadata['NWBFile']['session_start_time'] where the real date "
                "is known.",
                UserWarning,
                stacklevel=_stacklevel_outside_neuroconv(),
            )
        metadata["NWBFile"]["session_start_time"] = session_start_time
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
        write_annotations: bool = True,
        write_runs: bool = True,
    ):
        """
        Add the MEF 2 data to an NWBFile.

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
        write_annotations : bool, default: True
            If True and an annotations file was found or given, its annotations are written, in time order, to
            a ``TimeIntervals`` table named ``"annotations"``. If another interface has already written that
            table to the file with the same columns, the rows are added to it.
        write_runs : bool, default: True
            If True and the recording has more than one uninterrupted run, the runs are written to a
            ``TimeIntervals`` table named ``"runs"``, shared with MEF 3 interfaces, or added to it as for the
            annotations. When the directory holds several streams, the rows' ``source_file`` names the stream
            too, such as ``"recording (512 Hz)"``.
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
        if self.runs_selected and write_electrical_series:
            self._add_selected_runs_to_nwbfile(nwbfile=nwbfile, metadata=metadata, **conversion_options)
        else:
            super().add_to_nwbfile(nwbfile=nwbfile, metadata=metadata, **conversion_options)

        if write_annotations and self._annotations:
            self._add_annotations_to_nwbfile(nwbfile=nwbfile)
        # A single run is indistinguishable from an uninterrupted recording, so there is nothing to record,
        # unless it was selected from several: its index then says where it is in the recording.
        if write_runs and (self.runs_selected or len(self.mef2_recording.runs) > 1):
            self._add_runs_to_nwbfile(nwbfile=nwbfile)

    def _series_name_suffix(self, run_index: int) -> str:
        """The digits the writer appends to a run's series name in a file of the whole recording."""
        number_of_runs = len(self.mef2_recording.runs)
        if number_of_runs == 1:
            return ""
        return f"{run_index:0{int(math.ceil(math.log10(number_of_runs)))}}"

    def _add_selected_runs_to_nwbfile(self, *, nwbfile: NWBFile, metadata: dict | None, **conversion_options):
        """
        Write each selected run as its own series, named as in a file of the whole recording.

        The writer numbers the series of a recording by segment, which for a selection would be the run's
        place in the selection; each run is therefore written as a recording of one segment, under the name
        the whole recording gives it.
        """
        metadata = metadata or self._get_metadata_for_writing()
        selection = self.recording_extractor
        try:
            for position, run_index in enumerate(self.run_indices):
                self.recording_extractor = selection.select_segments([position])
                run_metadata = copy.deepcopy(metadata)
                entry = self._series_metadata_entry(run_metadata)
                entry["name"] = entry.get("name", "ElectricalSeries") + self._series_name_suffix(run_index)
                super().add_to_nwbfile(nwbfile=nwbfile, metadata=run_metadata, **conversion_options)
        finally:
            self.recording_extractor = selection

    def _series_metadata_entry(self, metadata: dict) -> dict:
        """The metadata entry ``add_to_nwbfile`` takes this interface's series name from, created if absent."""
        ecephys = metadata.setdefault("Ecephys", {})
        series = ecephys.get("ElectricalSeries")
        if isinstance(series, dict) and self.metadata_key in series:
            return series[self.metadata_key]
        return ecephys.setdefault(self.es_key, {})

    def _run_start_time(self, run_index: int) -> float:
        """Where run ``run_index`` of the recording starts on the NWB timeline, any alignment included."""
        if run_index in self.run_indices:
            return float(self.recording_extractor.get_start_time(segment_index=self.run_indices.index(run_index)))
        # A run left out of the selection, which an annotation can still fall in: its own start, moved as the
        # nearest selected run was moved by any alignment (the earlier one on a tie).
        nearest = min(self.run_indices, key=lambda index: (abs(index - run_index), index))
        shift = self._run_start_time(nearest) - self._raw_run_start_time(nearest)
        return self._raw_run_start_time(run_index) + shift

    def _raw_run_start_time(self, run_index: int) -> float:
        """Where run ``run_index`` starts on the recording's own timeline, before any alignment."""
        recording = self.mef2_recording
        return (recording.runs[run_index].start_time - recording.start_time) / 1e6

    def _session_time(self, time: int) -> float:
        """A µUTC time on the NWB timeline, from the start of its run as written, so any alignment moves it too."""
        run_index, seconds_into_run = self.mef2_recording.locate(time)
        return self._run_start_time(run_index) + seconds_into_run

    def _run_territories(self) -> list[tuple[float, float]]:
        """
        The stretch of the NWB timeline each run of the recording owns: from its start to the next run's,
        the first from the beginning of time and the last to its end.

        A gap belongs to the run before it, as :meth:`MEF2Recording.locate` places a time in a gap, and the
        stretches cover the timeline without overlap, so an annotation lands in at least one file of a
        recording split by runs, whatever gap it falls in.
        """
        starts = [self._run_start_time(index) for index in range(len(self.mef2_recording.runs))]
        return [
            (-math.inf if index == 0 else start, starts[index + 1] if index + 1 < len(starts) else math.inf)
            for index, start in enumerate(starts)
        ]

    def _add_annotations_to_nwbfile(self, nwbfile: NWBFile):
        sample_period = 1.0 / self.mef2_recording.sampling_frequency
        territories = self._run_territories() if self.runs_selected else None
        selected = set(self.run_indices)
        rows = []
        for annotation in self._annotations:
            start_time = self._session_time(annotation.start_time)
            # An annotation of an instant spans one sample, because NWB Inspector flags an interval that stops
            # where it starts.
            stop_time = max(self._session_time(annotation.end_time), start_time + sample_period)
            row = dict(
                start_time=start_time,
                stop_time=stop_time,
                label=annotation.annotation_type,
                annotation_description=annotation.description,
                layer=annotation.layer,
                annotator=annotation.annotator,
                channels=", ".join(annotation.channels),
            )
            if territories is not None:
                # Only what overlaps the selected runs' stretches; an annotation that reaches into a run written
                # elsewhere keeps its own times, and is flagged.
                overlapped = {
                    index for index, (low, high) in enumerate(territories) if start_time < high and stop_time > low
                }
                if not overlapped & selected:
                    continue
                row["spans_parts"] = bool(overlapped - selected)
            rows.append(row)
        if not rows:
            return
        columns = {
            # "label" rather than "description", which TimeIntervals reserves for the table's own description.
            "label": "The annotation's type, such as 'Seizure'.",
            "annotation_description": "The annotation's free-text description, empty if it has none.",
            "layer": "The IEEG.org annotation layer the annotation belongs to.",
            "annotator": "Who or what made the annotation.",
            "channels": "The channels the annotation applies to, by .mef file name, separated by ', '.",
        }
        if territories is not None:
            columns["spans_parts"] = (
                "Whether the annotation reaches past the runs this file holds, into runs of the recording written "
                "elsewhere; its start and stop times are its own, not cut to this file's."
            )
        add_rows_to_intervals_table(
            nwbfile=nwbfile,
            file_path=self.annotations_file_path,
            name="annotations",
            description="Annotations imported from an IEEG.org annotations.xml, placed on the samples they mark.",
            columns=columns,
            rows=rows,
            kind="MEF 2",
            source_file_description=_ANNOTATIONS_SOURCE_FILE_DESCRIPTION,
        )

    def _add_runs_to_nwbfile(self, nwbfile: NWBFile):
        recording = self.mef2_recording
        # The streams of a directory are converted by separate interfaces, each with runs of its own, so
        # each writes its rows once, named after its stream.
        source_file = f"{recording.path.name} ({recording.stream_name})" if recording.number_of_streams > 1 else None
        add_runs_to_nwbfile(
            nwbfile,
            run_start_times=[
                self.recording_extractor.get_start_time(segment_index=position)
                for position in range(len(self.run_indices))
            ],
            run_samples=[recording.runs[index].number_of_samples for index in self.run_indices],
            run_indices=self.run_indices,
            sampling_frequency=recording.sampling_frequency,
            file_path=recording.path,
            kind="MEF 2",
            source_file=source_file,
        )
