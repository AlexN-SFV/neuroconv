import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import DirectoryPath, FilePath
from pynwb import NWBFile

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
            the files are sampled at more than one rate; convert each stream with its own interface.
        annotations_file_path : str or Path, optional
            An IEEG.org ``annotations.xml`` for the recording. Defaults to ``annotations.xml`` in
            ``folder_path`` when there is one.
        validate_crc : bool, default: False
            If True, check each block's CRC as it is decoded and raise on a block or MEF 2.1 header that fails
            it, at a cost of a few percent of decoding time. Otherwise a failing header CRC only warns and block
            CRCs are not computed.
        """
        super().__init__(
            folder_path=folder_path,
            verbose=verbose,
            metadata_key=metadata_key,
            channels_to_skip=channels_to_skip,
            stream_name=stream_name,
            validate_crc=validate_crc,
        )
        if metadata_key is None:
            self.metadata_key = "mef2_recording"

        self.mef2_recording = self.recording_extractor.mef2_recording
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
            annotations.
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

        if write_annotations and self._annotations:
            self._add_annotations_to_nwbfile(nwbfile=nwbfile)
        # A single run is indistinguishable from an uninterrupted recording, so there is nothing to record.
        if write_runs and len(self.mef2_recording.runs) > 1:
            self._add_runs_to_nwbfile(nwbfile=nwbfile)

    def _session_time(self, time: int) -> float:
        """A µUTC time on the NWB timeline, from the start of its run as written, so any alignment moves it too."""
        run_index, seconds_into_run = self.mef2_recording.locate(time)
        return float(self.recording_extractor.get_start_time(segment_index=run_index)) + seconds_into_run

    def _add_annotations_to_nwbfile(self, nwbfile: NWBFile):
        sample_period = 1.0 / self.mef2_recording.sampling_frequency
        rows = []
        for annotation in self._annotations:
            start_time = self._session_time(annotation.start_time)
            stop_time = self._session_time(annotation.end_time)
            rows.append(
                dict(
                    start_time=start_time,
                    # An annotation of an instant spans one sample, because NWB Inspector flags an interval that
                    # stops where it starts.
                    stop_time=max(stop_time, start_time + sample_period),
                    label=annotation.annotation_type,
                    annotation_description=annotation.description,
                    layer=annotation.layer,
                    annotator=annotation.annotator,
                    channels=", ".join(annotation.channels),
                )
            )
        add_rows_to_intervals_table(
            nwbfile=nwbfile,
            file_path=self.annotations_file_path,
            name="annotations",
            description="Annotations imported from an IEEG.org annotations.xml, placed on the samples they mark.",
            columns={
                # "label" rather than "description", which TimeIntervals reserves for the table's own description.
                "label": "The annotation's type, such as 'Seizure'.",
                "annotation_description": "The annotation's free-text description, empty if it has none.",
                "layer": "The IEEG.org annotation layer the annotation belongs to.",
                "annotator": "Who or what made the annotation.",
                "channels": "The channels the annotation applies to, by .mef file name, separated by ', '.",
            },
            rows=rows,
            kind="MEF 2",
            source_file_description=_ANNOTATIONS_SOURCE_FILE_DESCRIPTION,
        )

    def _add_runs_to_nwbfile(self, nwbfile: NWBFile):
        rows = []
        for run_index, run in enumerate(self.mef2_recording.runs):
            start_time = float(self.recording_extractor.get_start_time(segment_index=run_index))
            rows.append(
                dict(
                    start_time=start_time,
                    stop_time=start_time + run.number_of_samples / self.mef2_recording.sampling_frequency,
                    run_index=run_index,
                )
            )
        # Keyed on the directory, so every interface over the recording writes its runs once. The streams of a
        # recording sampled at several rates are converted separately; only the first to reach the NWB file
        # writes its runs.
        add_rows_to_intervals_table(
            nwbfile=nwbfile,
            file_path=self.mef2_recording.path,
            name="runs",
            description=(
                "Uninterrupted runs of samples in recordings with gaps. Each run is written as its own "
                "ElectricalSeries, positioned at the run's start time."
            ),
            columns={"run_index": "0-based index of the uninterrupted run of samples within its recording."},
            rows=rows,
            kind="MEF 2",
            source_file_description=_RUNS_SOURCE_FILE_DESCRIPTION,
        )
