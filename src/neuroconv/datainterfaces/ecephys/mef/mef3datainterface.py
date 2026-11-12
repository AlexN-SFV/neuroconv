import copy
import warnings
from collections import Counter
from datetime import datetime, timezone
from typing import Literal

import numpy as np
from pydantic import DirectoryPath
from pynwb import NWBFile

from ._runs_table import add_runs_to_nwbfile
from .._channel_ids import _stacklevel_outside_neuroconv
from .._intervals_tables import add_rows_to_intervals_table
from ..baserecordingextractorinterface import BaseRecordingExtractorInterface
from ....utils import DeepDict, dict_deep_update

_SOURCE_FILE_DESCRIPTION = "Name of the MEF 3 .mefd session directory the row was read from."

# MEF writers de-identify a session by moving it to 1970-01-01; no real MEF 3 recording predates this.
_DEIDENTIFIED_BEFORE = datetime(1980, 1, 1, tzinfo=timezone.utc)


def _rate_label(rate: float, rates: tuple[float, ...], format_spec: str = "g") -> str:
    """
    A rate as the runs rows name it: ``1000`` for 1000.0 Hz, or the exact float when the short form would give
    two of the session's rates one label, since rows are written once per label and the second rate's would be
    lost. Series names and metadata keys use it with ``format_spec=".10g"``, so they never take an exponent and
    two rates never share a series.
    """
    if len({format(other, format_spec) for other in rates}) == len(set(rates)):
        return format(rate, format_spec)
    return str(float(rate))


class MEF3RecordingInterface(BaseRecordingExtractorInterface):
    """
    Data interface class for converting MEF 3.0 sessions (Multiscale Electrophysiology Format, ``.mefd``).

    The session is read by neuroconv's own reader, with a RED decoder that matches meflib, the reference
    implementation, bit for bit; neither pymef nor MNE is needed. The samples are decoded lazily, block by
    block, and stored as the integer counts the file holds, with each channel's units conversion factor as
    its gain.

    A session recorded with interruptions is written as one ``ElectricalSeries`` per uninterrupted run of
    blocks, sharing one electrodes table, each placed at its first block's start time, as discontinuous
    EDF+ is, or, with ``concatenate_runs_above``, a session with many runs as one series with timestamps.
    Each series uses the session's nominal sampling rate; a recording's clock drifts against it
    (by 28 ms over 55 minutes in ds004696), and the nominal rate is what puts the samples where BIDS event
    files, which count samples, place them.

    A session whose channels are sampled at several rates, or were not all recorded over the same stretches
    of time (a continuous channel beside channels with gaps, say), is read as several channel groups, each
    of one rate and one set of runs, and each written as its own series on the session's one timeline:
    ``ElectricalSeries_<rate>Hz``, with ``_<n>`` after it when several groups share a rate.
    :meth:`split_by_channel_group` gives one interface per group, for a ``ConverterPipe`` or to align or
    convert each on its own. A session with one group is written as before, as ``ElectricalSeries``.

    MEF records (notes, seizure and stimulation marks, ...) are written to a ``TimeIntervals`` table named
    ``"records"``, and the runs, when there is more than one, to one named ``"runs"``. Several sessions
    written into one NWB file share those tables, with a ``source_file`` column.

    A session whose metadata is encrypted is read with its password: section 2 is decrypted, and of section 3
    only the recording time offset and the GMT offset, never the subject's name, ID or recording location.
    Encrypted data blocks are refused.
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
    def get_available_channel_ids(
        folder_path: DirectoryPath,
        password: str | None = None,
        level_1_password: str | None = None,
        level_2_password: str | None = None,
    ) -> list:
        """
        Get the names of all time series channels in a MEF 3 session, reading one metadata file per channel.

        Parameters
        ----------
        folder_path : DirectoryPath
            Path to the session's ``.mefd`` directory.
        password, level_1_password, level_2_password : str, optional
            As for the interface, when the session's metadata is encrypted.

        Returns
        -------
        list
            The channel names, in acquisition channel order.
        """
        from ._mef3_crypto import MEF3Passwords
        from ._mef3_reader import available_channel_names

        return available_channel_names(
            folder_path, MEF3Passwords.from_arguments(password, level_1_password, level_2_password)
        )

    @staticmethod
    def get_channel_sampling_frequencies(
        folder_path: DirectoryPath,
        password: str | None = None,
        level_1_password: str | None = None,
        level_2_password: str | None = None,
    ) -> dict:
        """
        Get each channel's sampling frequency, reading one metadata file per channel.

        The interface writes each rate as its own series; this says which channels are at which rate without
        reading any index, to choose channels to leave out with ``channels_to_skip``.

        Parameters
        ----------
        folder_path : DirectoryPath
            Path to the session's ``.mefd`` directory.
        password, level_1_password, level_2_password : str, optional
            As for the interface, when the session's metadata is encrypted.

        Returns
        -------
        dict
            Channel name to sampling frequency in Hz, in acquisition channel order.
        """
        from ._mef3_crypto import MEF3Passwords
        from ._mef3_reader import channel_sampling_frequencies

        return channel_sampling_frequencies(
            folder_path, MEF3Passwords.from_arguments(password, level_1_password, level_2_password)
        )

    @staticmethod
    def get_empty_channel_ids(folder_path: DirectoryPath) -> list:
        """
        Get the names of the channels of a MEF 3 session that hold no samples, reading their block indices only.

        A writer may create a channel and never write a sample to it. The interface refuses such a channel unless
        it is left out; pass these to ``channels_to_skip`` to convert the rest. A skipped channel's records are
        kept, as any skipped channel's are. The indices are never encrypted, so no password is needed. A channel
        whose index cannot be read raises rather than being listed.

        Parameters
        ----------
        folder_path : DirectoryPath
            Path to the session's ``.mefd`` directory.

        Returns
        -------
        list
            The names of the channels that hold no samples, in name order; empty when every channel holds some.
        """
        from ._mef3_reader import empty_channel_names

        return empty_channel_names(folder_path)

    @classmethod
    def get_extractor_class(cls):
        from ._mef3_extractor import MEF3RecordingExtractor

        return MEF3RecordingExtractor

    def _initialize_extractor(self, interface_kwargs: dict):
        """
        Read the session once and build one recording per channel group, returning this interface's own.

        That is the first group's, or, for an interface from :meth:`split_by_channel_group`, its group's. The
        recordings are kept from the first call, made by ``__init__``; later calls (``get_original_timestamps``)
        only read the session again.
        """
        from ._mef3_crypto import MEF3Passwords
        from ._mef3_reader import read_mef3_session

        folder_path = interface_kwargs["folder_path"]
        channels_to_skip = interface_kwargs.get("channels_to_skip")
        concatenate_runs_above = interface_kwargs.get("concatenate_runs_above")
        self.extractor_kwargs = dict(folder_path=folder_path, channels_to_skip=channels_to_skip)
        # The passwords go to the reader but not into extractor_kwargs, which describe the extractor to others.
        passwords = MEF3Passwords.from_arguments(
            *(interface_kwargs.get(key) for key in ("password", "level_1_password", "level_2_password"))
        )
        session = read_mef3_session(folder_path, channels_to_skip=channels_to_skip, passwords=passwords)

        extractor_class = self.get_extractor_class()
        extractors = []
        for index in range(len(session.channel_groups)):
            group_session = session.select_group(index) if len(session.channel_groups) > 1 else session
            kept = {channel.name for channel in group_session.channels}
            group_channels_to_skip = list(channels_to_skip or []) + [
                channel.name for channel in session.channels if channel.name not in kept
            ]
            runs = session.channel_groups[index].runs
            concatenate_runs = concatenate_runs_above is not None and len(runs) > concatenate_runs_above
            extractors.append(
                extractor_class(
                    folder_path=folder_path,
                    channels_to_skip=group_channels_to_skip or None,
                    concatenate_runs=concatenate_runs,
                    _session=group_session,
                )
            )
        if not hasattr(self, "_group_extractors"):
            self._group_extractors = extractors
            self._channel_group_interfaces = None
            self.mef3_session = session
        return extractors[getattr(self, "_channel_group_index", 0)]

    def __init__(
        self,
        folder_path: DirectoryPath,
        *,
        verbose: bool = False,
        metadata_key: str | None = None,
        channels_to_skip: list | None = None,
        password: str | None = None,
        level_1_password: str | None = None,
        level_2_password: str | None = None,
        concatenate_runs_above: int | None = None,
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
        password : str, optional
            The session's password, when its metadata is encrypted; its level (1 or 2) is found from the files.
            A level 2 password opens both levels. Ignored for a session that is not encrypted. Take it from a
            secret or an environment variable. No message or warning names it, and the extractor's kwargs leave
            it out.
        level_1_password, level_2_password : str, optional
            The session's passwords by level, instead of ``password``; each is checked at its own level.
        concatenate_runs_above : int, optional
            Write a channel group with more than this many runs as one ``ElectricalSeries`` with timestamps,
            the runs back to back and the gaps in the timestamps, instead of one series per run. The default,
            None, always writes one series per run. Every series costs time to build and write and space in the
            file whatever its length, so a recording with thousands of short runs (a recorder that drops a few
            samples every so often) takes far longer as thousands of series than as one, which the timestamps
            make about twice the size of the samples for a single channel. The ``"runs"`` table lists the runs
            either way.
        """
        super().__init__(
            folder_path=folder_path,
            verbose=verbose,
            metadata_key=metadata_key,
            channels_to_skip=channels_to_skip,
            password=password,
            level_1_password=level_1_password,
            level_2_password=level_2_password,
            concatenate_runs_above=concatenate_runs_above,
        )
        if metadata_key is None:
            self.metadata_key = "mef3_recording"

    @property
    def _has_several_groups(self) -> bool:
        return len(self._group_extractors) > 1

    def split_by_channel_group(self) -> list["MEF3RecordingInterface"]:
        """
        One interface per channel group of the session: channels of one rate, recorded over the same runs.

        A session with one group gives ``[self]``. Otherwise each interface holds one group's recording and is
        given a distinct ``metadata_key`` and series name, ``<metadata_key>_<rate>Hz`` and
        ``ElectricalSeries_<rate>Hz``, followed by ``_<n>`` when several groups share a rate, so they can be
        written through one ``ConverterPipe``, or aligned and converted one by one. Every group is on the
        session's timeline and carries its records, which only the first to reach a file writes; each group's
        runs are labelled with its rate, and its number when several share it, in the ``"runs"`` table.

        The interfaces are made once, from this interface's keys at the time, and share its recordings, so an
        alignment applied to one applies wherever its recording is used.

        Returns
        -------
        list of MEF3RecordingInterface
            The largest group first: the most channels, then the highest rate.
        """
        if not self._has_several_groups:
            return [self]
        if self._channel_group_interfaces is None:
            groups = self.mef3_session.channel_groups
            groups_at_rate = Counter(group.sampling_frequency for group in groups)
            numbered_at_rate = Counter()
            series_name = getattr(self, "_split_series_name", None) or "ElectricalSeries"
            interfaces = []
            for index, (group, extractor) in enumerate(zip(groups, self._group_extractors)):
                rate = group.sampling_frequency
                rates = self.mef3_session.session_sampling_frequencies
                suffix = f"{_rate_label(rate, rates, '.10g')}Hz"
                label = f"{_rate_label(rate, rates)} Hz"
                if groups_at_rate[rate] > 1:
                    number = numbered_at_rate[rate]
                    numbered_at_rate[rate] += 1
                    # Zero-padded so the names stay distinct after neuroconv appends a run's index to them.
                    suffix += f"_{number:0{len(str(groups_at_rate[rate] - 1))}}"
                    label += f", group {number}"
                interface = copy.copy(self)
                interface.recording_extractor = extractor
                interface._number_of_segments = extractor.get_num_segments()
                interface._group_extractors = [extractor]
                interface._channel_group_interfaces = None
                interface._channel_group_index = index
                interface._runs_label = label
                interface.mef3_session = extractor.mef3_session
                interface.metadata_key = f"{self.metadata_key}_{suffix}"
                interface.es_key = f"{self.es_key}_{suffix}"
                interface._split_series_name = f"{series_name}_{suffix}"
                interfaces.append(interface)
            self._channel_group_interfaces = interfaces
        return list(self._channel_group_interfaces)

    def split_by_offset(self) -> list[BaseRecordingExtractorInterface]:
        """
        Split by channel group (see :meth:`split_by_channel_group`), then each group by channel offset.

        MEF 3 stores no offset, so a session's parts are its channel groups unless an offset has been set.
        """
        if not self._has_several_groups:
            return super().split_by_offset()
        # Each group's parts keep the group's name ahead of the offset's index, so parts of different groups
        # do not collide.
        return [
            part
            for group_interface in self.split_by_channel_group()
            for part in BaseRecordingExtractorInterface.split_by_offset(group_interface)
        ]

    def _refuse_several_groups(self, action: str):
        if self._has_several_groups:
            raise NotImplementedError(
                f"The MEF 3 session '{self.mef3_session.path.name}' is read as "
                f"{len(self._group_extractors)} channel groups, each with its own rate or runs, so it cannot "
                f"{action} as one recording. Use split_by_channel_group() and {action} for each group."
            )

    @property
    def channel_ids(self):
        """Gets the channel ids of the data: every group's, in the session's channel order."""
        if self._has_several_groups:
            return [channel.name for channel in self.mef3_session.channels]
        return super().channel_ids

    def remove_channels(self, channel_ids: list):
        self._refuse_several_groups("remove channels")
        return super().remove_channels(channel_ids=channel_ids)

    def get_original_timestamps(self):
        self._refuse_several_groups("give timestamps")
        return super().get_original_timestamps()

    def get_timestamps(self):
        self._refuse_several_groups("give timestamps")
        return super().get_timestamps()

    def set_aligned_timestamps(self, aligned_timestamps):
        self._refuse_several_groups("align timestamps")
        super().set_aligned_timestamps(aligned_timestamps=aligned_timestamps)

    def set_aligned_segment_timestamps(self, aligned_segment_timestamps):
        self._refuse_several_groups("align timestamps")
        super().set_aligned_segment_timestamps(aligned_segment_timestamps=aligned_segment_timestamps)

    def set_aligned_segment_starting_times(self, aligned_segment_starting_times):
        self._refuse_several_groups("align runs")
        super().set_aligned_segment_starting_times(aligned_segment_starting_times=aligned_segment_starting_times)

    def align_by_interpolation(self, unaligned_timestamps, aligned_timestamps):
        self._refuse_several_groups("align timestamps")
        super().align_by_interpolation(unaligned_timestamps=unaligned_timestamps, aligned_timestamps=aligned_timestamps)

    def set_probe(self, probe, group_mode):
        self._refuse_several_groups("take a probe")
        super().set_probe(probe=probe, group_mode=group_mode)

    def set_aligned_starting_time(self, aligned_starting_time: float):
        """Shift every channel group by ``aligned_starting_time`` seconds, keeping them on one timeline."""
        for interface in self.split_by_channel_group():
            if interface is self:
                super().set_aligned_starting_time(aligned_starting_time=aligned_starting_time)
            else:
                interface.set_aligned_starting_time(aligned_starting_time=aligned_starting_time)

    def get_metadata(self, *, use_new_metadata_format: bool = True) -> DeepDict:
        if self._has_several_groups:
            # Every group's series entry; the rest is the session's, the same from every group.
            metadata = None
            for interface in self.split_by_channel_group():
                group_metadata = interface.get_metadata(use_new_metadata_format=use_new_metadata_format)
                metadata = group_metadata if metadata is None else dict_deep_update(metadata, group_metadata)
            return metadata

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

        A session read as several channel groups writes each group as its own series, with these options.
        """
        if self._has_several_groups:
            for interface in self.split_by_channel_group():
                interface.add_to_nwbfile(
                    nwbfile=nwbfile,
                    metadata=metadata,
                    stub_test=stub_test,
                    write_as=write_as,
                    parent_container=parent_container,
                    data_representation=data_representation,
                    write_electrical_series=write_electrical_series,
                    iterator_type=iterator_type,
                    iterator_options=iterator_options,
                    always_write_timestamps=always_write_timestamps,
                    write_records=write_records,
                    write_runs=write_runs,
                )
            return

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

    def _run_start_times(self) -> np.ndarray:
        """
        Each run's start on the session timeline, read from the recording, so any alignment moves the tables too.

        A run is a segment of the recording, or, when the runs were concatenated, a stretch of its one segment.
        """
        recording = self.recording_extractor
        runs = self.mef3_session.runs
        if recording.get_num_segments() == len(runs):
            return np.array([recording.get_start_time(segment_index=index) for index in range(len(runs))], dtype=float)
        first_samples = np.concatenate([[0], np.cumsum([run.number_of_samples for run in runs])[:-1]])
        return np.asarray(recording.get_times(segment_index=0)[first_samples], dtype=float)

    def _session_time(self, time: int, run_start_times: np.ndarray | None = None) -> float:
        """
        A µUTC time on the session timeline, shifted with the run it falls in, or the run before it.

        ``run_start_times`` is :meth:`_run_start_times`, passed in when placing many times.
        """
        runs = self.mef3_session.runs
        if run_start_times is None:
            run_start_times = self._run_start_times()
        run_index = max(int(np.searchsorted([run.start_time for run in runs], time, side="right")) - 1, 0)
        run = runs[run_index]
        # The run's shift from where the reader put it: 0.0 exactly unless the recording was aligned. Adding it
        # to the time from the session's start, rather than going through the run's own start, gives the
        # same float for a record whichever interface over the session writes it.
        session_start = self.mef3_session.start_time
        shift = float(run_start_times[run_index]) - (run.start_time - session_start) / 1e6
        return (time - session_start) / 1e6 + shift

    def _add_records_to_nwbfile(self, nwbfile: NWBFile):
        from ._mef3_reader import UUTC_NO_ENTRY

        # The records table is the session's and is written once, by whichever interface over the session
        # comes first, so an instant spans one sample of the session's fastest rate, not of that interface's.
        sample_period = 1.0 / max(self.mef3_session.session_sampling_frequencies)
        rows = []
        # A record whose own time is "no entry" states no time to place it at, so it is left out.
        untimed = [record for record in self.mef3_session.records if record.time == UUTC_NO_ENTRY]
        if untimed:
            warnings.warn(
                f"{len(untimed)} records of the MEF 3 session '{self.mef3_session.path.name}' state no time "
                "(UUTC_NO_ENTRY) and are left out of the records table.",
                UserWarning,
                stacklevel=_stacklevel_outside_neuroconv(),
            )
        run_start_times = self._run_start_times()
        for record in self.mef3_session.records:
            if record.time == UUTC_NO_ENTRY:
                continue
            start_time = self._session_time(record.time, run_start_times)
            # The other times the record states are placed on the same timeline, in seconds, so that no absolute
            # time is written and every time in the file stays relative to session_start_time. A stated time
            # before the session started is not a time of this recording (a placeholder, or the clock of another
            # one) and would only write the session's own start back, negated, so it is left out.
            details = "; ".join(
                part
                for part in [record.details]
                + [
                    f"{name}_s={self._session_time(value, run_start_times):.6f}"
                    for name, value in record.times.items()
                    if value >= self.mef3_session.start_time
                ]
                if part
            )
            rows.append(
                dict(
                    start_time=start_time,
                    # A record without a duration marks an instant; it spans one sample, because NWB Inspector
                    # flags an interval that stops where it starts.
                    stop_time=start_time + (record.duration / 1e6 if record.duration > 0 else sample_period),
                    record_type=record.record_type,
                    label=record.label,
                    details=details,
                    duration_in_microseconds=record.duration,
                    channel=record.channel,
                )
            )
        if not rows:
            return
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
                "details": (
                    "The record's other fields, as 'name=value' pairs separated by '; '. A time the record states "
                    "(a seizure's earliest onset, a cursor's trace time) is named with '_s' and given in seconds on "
                    "the same timeline as start_time."
                ),
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
        session = self.mef3_session
        # A session at several rates, or in several channel groups, is written a group at a time, whose runs
        # differ; each group's rows are written, labelled with its rate as MEF 2 streams are, and its number when
        # several groups share the rate. The records are the session's, written once.
        label = getattr(self, "_runs_label", None)
        if label is None and len(session.session_sampling_frequencies) > 1:
            label = f"{_rate_label(session.sampling_frequency, session.session_sampling_frequencies)} Hz"
        source_file = f"{session.path.name} ({label})" if label else None
        add_runs_to_nwbfile(
            nwbfile,
            run_start_times=self._run_start_times(),
            run_samples=[run.number_of_samples for run in session.runs],
            sampling_frequency=session.sampling_frequency,
            file_path=session.path,
            kind="MEF 3",
            source_file=source_file,
        )
