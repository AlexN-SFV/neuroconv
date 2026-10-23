import warnings
from datetime import datetime, timezone
from typing import Literal

from pydantic import DirectoryPath
from pynwb import NWBFile

from ._runs_table import add_runs_to_nwbfile
from .._channel_ids import _stacklevel_outside_neuroconv
from .._intervals_tables import add_rows_to_intervals_table
from ..baserecordingextractorinterface import BaseRecordingExtractorInterface
from ....utils import DeepDict

_SOURCE_FILE_DESCRIPTION = "Name of the MEF 3 .mefd session directory the row was read from."

# MEF writers de-identify a session by moving it to 1970-01-01; no real MEF 3 recording predates this.
_DEIDENTIFIED_BEFORE = datetime(1980, 1, 1, tzinfo=timezone.utc)


class MEF3RecordingInterface(BaseRecordingExtractorInterface):
    """
    Data interface class for converting MEF 3.0 sessions (Multiscale Electrophysiology Format, ``.mefd``).

    The session is read by neuroconv's own reader, with a RED decoder that matches meflib, the reference
    implementation, bit for bit; neither pymef nor MNE is needed. The samples are decoded lazily, block by
    block, and stored as the integer counts the file holds, with each channel's units conversion factor as
    its gain.

    A session recorded with interruptions is written as one ``ElectricalSeries`` per uninterrupted run of
    blocks, sharing one electrodes table, each placed at its first block's start time, as discontinuous
    EDF+ is. Each series uses the session's nominal sampling rate; a recording's clock drifts against it
    (by 28 ms over 55 minutes in ds004696), and the nominal rate is what puts the samples where BIDS event
    files, which count samples, place them.

    MEF records (notes, seizure and stimulation marks, ...) are written to a ``TimeIntervals`` table named
    ``"records"``, and the runs, when there is more than one, to one named ``"runs"``. Several sessions
    written into one NWB file share those tables, with a ``source_file`` column.

    Encrypted sessions are refused.
    """

    display_name = "MEF 3 Recording"
    keywords = BaseRecordingExtractorInterface.keywords + (
        "MEF",
        "MEF3",
        "Multiscale Electrophysiology Format",
        "iEEG",
        "EEG",
    )
    associated_suffixes = (".mefd",)
    info = "Interface for MEF 3.0 (Multiscale Electrophysiology Format) session recording data."

    @classmethod
    def get_source_schema(cls) -> dict:
        source_schema = super().get_source_schema()
        source_schema["properties"]["folder_path"]["description"] = "Path to the session's .mefd directory."
        return source_schema

    @staticmethod
    def get_available_channel_ids(folder_path: DirectoryPath) -> list:
        """
        Get the names of all time series channels in a MEF 3 session, reading one metadata file per channel.

        Parameters
        ----------
        folder_path : DirectoryPath
            Path to the session's ``.mefd`` directory.

        Returns
        -------
        list
            The channel names, in acquisition channel order.
        """
        from ._mef3_reader import available_channel_names

        return available_channel_names(folder_path)

    @classmethod
    def get_extractor_class(cls):
        from ._mef3_extractor import MEF3RecordingExtractor

        return MEF3RecordingExtractor

    def _initialize_extractor(self, interface_kwargs: dict):
        self.extractor_kwargs = dict(
            folder_path=interface_kwargs["folder_path"],
            channels_to_skip=interface_kwargs.get("channels_to_skip"),
        )
        return self.get_extractor_class()(**self.extractor_kwargs)

    def __init__(
        self,
        folder_path: DirectoryPath,
        *,
        verbose: bool = False,
        metadata_key: str | None = None,
        channels_to_skip: list | None = None,
    ):
        """
        Load and prepare data for MEF 3.

        Parameters
        ----------
        folder_path : str or Path
            Path to the session's ``.mefd`` directory.
        verbose : bool, default: False
            Allows verbose.
        metadata_key : str, optional
            Key that indexes this interface's entries in the dict-based metadata. Defaults to
            ``"mef3_recording"``.
        channels_to_skip : list, default: None
            Names of channels to leave out, such as auxiliary channels that are not neural data, or channels
            sampled at a rate the others are not.
        """
        super().__init__(
            folder_path=folder_path,
            verbose=verbose,
            metadata_key=metadata_key,
            channels_to_skip=channels_to_skip,
        )
        if metadata_key is None:
            self.metadata_key = "mef3_recording"

        self.mef3_session = self.recording_extractor.mef3_session

    def get_metadata(self, *, use_new_metadata_format: bool = True) -> DeepDict:
        metadata = super().get_metadata(use_new_metadata_format=use_new_metadata_format)
        session_start_time = self.mef3_session.start_datetime
        if session_start_time < _DEIDENTIFIED_BEFORE:
            warnings.warn(
                f"The MEF 3 session '{self.mef3_session.path.name}' starts at {session_start_time.isoformat()}, "
                "the date MEF writers move a de-identified session to, not when it was recorded. It is used as "
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
        write_records: bool = True,
        write_runs: bool = True,
    ):
        """
        Add the MEF 3 data to an NWBFile.

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
        write_records : bool, default: True
            If True and the session holds MEF records, they are written, in time order, to a ``TimeIntervals``
            table named ``"records"``. If another MEF 3 interface has already written that table to the file,
            the rows are added to it.
        write_runs : bool, default: True
            If True and the session has more than one uninterrupted run, the runs are written to a
            ``TimeIntervals`` table named ``"runs"``, or added to it as for the records.
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

        if write_records and self.mef3_session.records:
            self._add_records_to_nwbfile(nwbfile=nwbfile)
        # A single run is indistinguishable from an uninterrupted recording, so there is nothing to record.
        if write_runs and len(self.mef3_session.runs) > 1:
            self._add_runs_to_nwbfile(nwbfile=nwbfile)

    def _run_start_time(self, run_index: int) -> float:
        """A run's start on the session timeline, read from the recording, so any alignment moves the tables too."""
        return float(self.recording_extractor.get_start_time(segment_index=run_index))

    def _session_time(self, time: int) -> float:
        """A µUTC time on the session timeline, shifted with the run it falls in, or the run before it."""
        runs = self.mef3_session.runs
        run_index = 0
        for index, run in enumerate(runs):
            if time >= run.start_time:
                run_index = index
        run = runs[run_index]
        return self._run_start_time(run_index) + (time - run.start_time) / 1e6

    def _add_records_to_nwbfile(self, nwbfile: NWBFile):
        sample_period = 1.0 / self.mef3_session.sampling_frequency
        rows = []
        for record in self.mef3_session.records:
            start_time = self._session_time(record.time)
            rows.append(
                dict(
                    start_time=start_time,
                    # A record without a duration marks an instant; it spans one sample, because NWB Inspector
                    # flags an interval that stops where it starts.
                    stop_time=start_time + (record.duration / 1e6 if record.duration > 0 else sample_period),
                    record_type=record.record_type,
                    label=record.label,
                    details=record.details,
                    duration_in_microseconds=record.duration,
                    channel=record.channel,
                )
            )
        add_rows_to_intervals_table(
            nwbfile=nwbfile,
            file_path=self.mef3_session.path,
            name="records",
            description="Records imported from MEF 3 sessions (.rdat/.ridx), at session, channel and segment level.",
            columns={
                "record_type": "The record's four-letter MEF type, such as 'Note', 'Seiz', 'EDFA' or 'ESti'.",
                # "label" rather than "description", which TimeIntervals reserves for the table's own description.
                "label": (
                    "The record's text: the note of a Note or SyLg, the annotation of an EDFA or Seiz, the stimulus "
                    "type of a CSti, the waveform of an ESti, the name of a Curs, the text of an Epoc."
                ),
                "details": "The record's other fields, as 'name=value' pairs separated by '; '.",
                "duration_in_microseconds": (
                    "The duration the record states, 0 for a record without one, which spans one sample here."
                ),
                "channel": "The channel a channel- or segment-level record belongs to, empty for a session-level one.",
            },
            rows=rows,
            kind="MEF 3",
            source_file_description=_SOURCE_FILE_DESCRIPTION,
        )

    def _add_runs_to_nwbfile(self, nwbfile: NWBFile):
        add_runs_to_nwbfile(
            nwbfile,
            recording_extractor=self.recording_extractor,
            run_samples=[run.number_of_samples for run in self.mef3_session.runs],
            sampling_frequency=self.mef3_session.sampling_frequency,
            file_path=self.mef3_session.path,
            kind="MEF 3",
        )
