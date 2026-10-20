import re
import warnings
from datetime import date, datetime
from pathlib import Path
from typing import Literal

from pydantic import FilePath
from pynwb import NWBFile

from ..baserecordingextractorinterface import BaseRecordingExtractorInterface
from ....tools import get_package
from ....utils import DeepDict

# Month abbreviations are matched against this table rather than through ``strptime("%b")``, which
# resolves them via LC_TIME and would silently fail to parse an English month name under another
# locale — dropping the birthdate on machines that are configured differently.
_MONTH_ABBREVIATIONS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]


def _warn_unreadable(field: str, value) -> None:
    """
    Warn that a subject field was present in the header but could not be interpreted.

    "The file did not state this" and "the file stated something I could not read" are different
    situations, and only the second is the user's to act on — otherwise an exporter writing an
    unexpected format produces a silently subject-less NWB file with no indication why.
    """
    # stacklevel=3 lands on extract_subject_metadata, the innermost public caller. No fixed value can
    # point at the user's own line, because both extract_subject_metadata and get_metadata are public
    # entry points and sit at different depths; the offending value is in the message, which is the
    # actionable part.
    warnings.warn(
        f"The EDF header's {field} could not be interpreted and was left out of the NWB Subject: "
        f"{value!r}. Set metadata['Subject'] explicitly if this field matters for your conversion.",
        UserWarning,
        stacklevel=3,
    )


# Floor on a plausible birth year. This is a judgement call rather than a derived constraint: it only
# has to be low enough never to reject a real subject and high enough to catch a misread field, and
# 1850 clears both — EDF itself dates from 1992, so no subject of an EDF recording was born before it,
# while "02-MAY-0001" is plainly garbage.
_EARLIEST_PLAUSIBLE_BIRTH_YEAR = 1850


def _assemble_birthdate(year: int, month: int, day: int, tzinfo=None) -> datetime | None:
    """Build a birthdate, rejecting impossible calendar dates and implausible years."""
    if year < _EARLIEST_PLAUSIBLE_BIRTH_YEAR:
        return None
    try:
        birthdate = datetime(year=year, month=month, day=day, tzinfo=tzinfo)
    except ValueError:
        return None
    # A birthdate cannot be in the future; such a value means the field was misread, not that the
    # subject is unborn, and passing it on would put a nonsense date in the NWB file.
    now = datetime.now(tz=birthdate.tzinfo)
    return None if birthdate > now else birthdate


def _parse_birthdate(birthdate) -> datetime | None:
    """
    Coerce an EDF birthdate into a datetime, or None when it cannot be read.

    NWB's ``Subject.date_of_birth`` must be a datetime, but the readers hand the EDF+ patient field's
    birthdate back as a string — pyedflib normalizes it to ``"02 may 1951"`` — so an absent or
    unreadable value is dropped rather than passed on to pynwb.

    Accepted forms are the ``DD-MMM-YYYY`` the EDF+ spec defines for this field (in any case, and
    space- or hyphen-separated, which covers what pyedflib emits as well as a direct header read) and
    ISO ``YYYY-MM-DD``. Notably ``DD.MM.YY`` is *not*: no reader emits it for a birthdate — the
    two-digit form belongs to the header's separate ``startdate`` field — and Python's POSIX rule for
    ``%y`` maps 00-68 to the 2000s, so ``"02.05.51"`` would have become 2051, a date in the future.
    """
    # An absent field, or one that states "unknown", is not a problem and stays silent. Only a value
    # that is *present* and unreadable is worth telling the user about: this whole change exists because
    # metadata was being discarded without a trace, and dropping a birthdate the exporter did write
    # would repeat that in miniature.
    #
    # In practice both readers normalize EDF+'s "X" marker to "" before it reaches here, so the
    # unknown-marker check is defensive; it keeps this helper agreeing with _parse_sex about what counts
    # as a declaration rather than a failure, should a third caller ever pass the raw subfield through.
    if birthdate is None:
        return None
    if not isinstance(birthdate, date) and str(birthdate).strip().upper() in _VALUES_MEANING_UNKNOWN:
        return None

    parsed = None
    if isinstance(birthdate, date):
        # datetime is a subclass of date, so this covers both. The time of day is dropped because a
        # birthdate has none; any tzinfo is preserved rather than silently discarded.
        parsed = _assemble_birthdate(
            birthdate.year, birthdate.month, birthdate.day, tzinfo=getattr(birthdate, "tzinfo", None)
        )
    else:
        text = str(birthdate).strip()
        named_month = re.match(r"^(\d{1,2})[-\s]([A-Za-z]{3,})[-\s](\d{4})$", text)
        abbreviation = named_month.group(2)[:3].upper() if named_month else None
        if abbreviation in _MONTH_ABBREVIATIONS:
            parsed = _assemble_birthdate(
                year=int(named_month.group(3)),
                month=_MONTH_ABBREVIATIONS.index(abbreviation) + 1,
                day=int(named_month.group(1)),
            )
        elif iso := re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", text):
            parsed = _assemble_birthdate(year=int(iso.group(1)), month=int(iso.group(2)), day=int(iso.group(3)))

    if parsed is None:
        _warn_unreadable(field="date of birth", value=birthdate)
    return parsed


# EDF+ mandates F/M for the sex subfield and pyedflib normalizes it to Female/Male, so an exact
# allowlist covers everything any reader emits. Prefix matching would be looser than the input space
# and can invert the value outright — "mujer" starts with an M — which is the very failure this
# extraction exists to avoid.
_SEX_BY_HEADER_VALUE = {
    "F": "F",
    "FEMALE": "F",
    "M": "M",
    "MALE": "M",
    "O": "O",
    "OTHER": "O",
}

# Ways of saying "unknown". These are declarations, not failures: the file stated the field perfectly
# clearly, as unknown, so warning that it "could not be interpreted" would report a malformed file where
# there is none. EDF+ uses "X" for any unknown subfield, birthdate included, which is why this is shared
# between both parsers rather than specific to sex.
_VALUES_MEANING_UNKNOWN = ("", "X", "UNKNOWN", "N/A", "NA", "?", "-")

# NWB's own code for unknown sex is "U" — the very value the interface falls back to — so it belongs in
# the silent set too, but only for sex, where it means something.
_SEX_VALUES_MEANING_UNKNOWN = _VALUES_MEANING_UNKNOWN + ("U",)


def _parse_sex(sex) -> str | None:
    """
    Map an EDF+ sex subfield onto the letter NWB expects, or None when it is not stated.

    Anything outside the allowlist is treated as not stated, so the caller's ``"U"`` default applies
    rather than a guess. An empty field and EDF+'s bare ``X`` mean "unknown" and pass silently; any
    other unrecognized value warns, since the file did state something.
    """
    if sex is None:
        return None
    # Note: not ``str(sex or "")`` — that would fold a numeric 0 into the silent path, hiding the one
    # input shape the allowlist cannot interpret.
    text = str(sex).strip().upper()
    if text in _SEX_VALUES_MEANING_UNKNOWN:
        return None
    mapped = _SEX_BY_HEADER_VALUE.get(text)
    if mapped is None:
        _warn_unreadable(field="subject sex", value=sex)
    return mapped


class EDFRecordingInterface(BaseRecordingExtractorInterface):
    """
    Data interface class for converting European Data Format (EDF) data.

    Plain EDF and continuous EDF+ (``EDF+C``) files are read with the
    :py:func:`~spikeinterface.extractors.read_edf` reader from SpikeInterface.

    Discontinuous EDF+ (``EDF+D``) files are detected from the header and read with neuroconv's own
    reader instead, because neither SpikeInterface's reader nor the ``pyedflib`` library beneath it can
    open them at all. Each contiguous run of data records becomes one segment positioned at its true
    start time on the session timeline, so a file whose records are in fact contiguous — common, since
    exporters label continuous recordings ``EDF+D`` as a matter of course — is written as a single
    ``ElectricalSeries`` exactly like a plain EDF.

    Signals carrying floating-point or long-integer data through the EDF+ logarithmic transformation
    (physical dimension ``Filtered``) are decoded either way, through one shared implementation: on the
    SpikeInterface path by wrapping its reader, which has no notion of the transformation and would
    otherwise apply the header's linear gain to what are logarithms, and natively in neuroconv's own
    reader. See https://www.edfplus.info/specs/edffloat.html.

    The EDF/EDF+C path is not supported on M1 macs, because ``pyedflib`` is not available there. The
    EDF+D path has no such restriction — it needs only NumPy.
    """

    display_name = "EDF Recording"
    keywords = BaseRecordingExtractorInterface.keywords + ("European Data Format",)
    associated_suffixes = (".edf",)
    info = "Interface for European Data Format (EDF) recording data."

    @classmethod
    def get_source_schema(cls) -> dict:
        source_schema = super().get_source_schema()
        source_schema["properties"]["file_path"]["description"] = "Path to the .edf file."
        return source_schema

    @classmethod
    def get_stream_names(cls, file_path: FilePath) -> list[str]:
        """
        Get the names of the streams available in an EDF file.

        A stream is a set of channels that share a sampling rate, so a file that sampled some of
        its signals at a different rate than the rest carries more than one. Streams apply to plain
        EDF and ``EDF+C`` files only; neuroconv's ``EDF+D`` reader has no notion of them.

        Parameters
        ----------
        file_path : FilePath
            Path to the EDF file

        Returns
        -------
        list of str
            List of the stream names in the EDF file
        """
        from spikeinterface.extractors.extractor_classes import EDFRecordingExtractor

        stream_names, _ = EDFRecordingExtractor.get_streams(file_path=file_path)
        return stream_names

    @staticmethod
    def get_available_channel_ids(file_path: FilePath) -> list:
        """
        Get all available channel names from an EDF file.

        The names span the whole file. A file that sampled some of its signals at a different rate
        than the rest holds them in separate streams, and an interface reads one stream at a time, so
        the channels of the stream it holds are a subset of these. They are read from the file's
        header, so this works on a file with more than one stream.

        Parameters
        ----------
        file_path : FilePath
            Path to the EDF file

        Returns
        -------
        list
            List of all channel names in the EDF file
        """
        from ._edfd_reader import is_discontinuous_edf, read_edf_header

        if is_discontinuous_edf(file_path=file_path):
            # pyedflib cannot open EDF+D at all; read the labels out of the header directly. The
            # annotations signals are left out, as pyedflib leaves them out — they are never data
            # channels, so listing them would only invite a pointless channels_to_skip entry.
            with open(file_path, "rb") as file:
                header = read_edf_header(file)
            annotations_indices = set(header.annotations_signal_indices)
            return [label for index, label in enumerate(header.labels) if index not in annotations_indices]

        from pyedflib import EdfReader

        edf_reader = EdfReader(str(file_path))
        try:
            channel_names = edf_reader.getSignalLabels()
        finally:
            # EDFlib refuses to open a file it already has open, so the handle is released here
            # rather than left to garbage collection.
            edf_reader.close()

        return channel_names

    @classmethod
    def get_extractor_class(cls):
        from spikeinterface.extractors.extractor_classes import EDFRecordingExtractor

        return EDFRecordingExtractor

    @classmethod
    def get_discontinuous_extractor_class(cls):
        """Return neuroconv's own extractor, used for discontinuous EDF+ (``EDF+D``) files."""
        from ._edfd_extractor import EDFDRecordingExtractor

        return EDFDRecordingExtractor

    def _initialize_extractor(self, interface_kwargs: dict):
        """Route discontinuous EDF+ to neuroconv's reader; otherwise read with SpikeInterface."""
        from ._edfd_reader import is_discontinuous_edf

        # Discontinuity is the one property SpikeInterface's reader cannot represent at all — it refuses
        # such files, and models every file as a single contiguous segment. A logarithmically transformed
        # signal, by contrast, only needs decoding on top of that reader, which is what the continuous
        # path below does; so only EDF+D is routed away.
        self._is_discontinuous = is_discontinuous_edf(file_path=interface_kwargs["file_path"])
        if self._is_discontinuous:
            if interface_kwargs.get("stream_name") is not None:
                raise ValueError(
                    "stream_name applies only to plain EDF and EDF+C files, which are read by SpikeInterface. "
                    f"'{interface_kwargs['file_path']}' is a discontinuous EDF+ (EDF+D) file, read by neuroconv's "
                    "own reader, which has no streams; drop the channels at another sampling rate with "
                    "channels_to_skip instead."
                )
            # channels_to_skip is applied while reading here rather than afterwards, so a channel at a
            # differing sampling rate — or a transformed signal whose parameters are unreadable — can be
            # dropped before it stops the conversion.
            self.extractor_kwargs = dict(
                file_path=interface_kwargs["file_path"],
                channels_to_skip=interface_kwargs.get("channels_to_skip"),
            )
            recording = self.get_discontinuous_extractor_class()(**self.extractor_kwargs)

            from ._edfd_reader import edf_plus_header_fields

            # Presented in the shape pyedflib returns, so the metadata extraction is identical either way.
            self._edf_header = edf_plus_header_fields(recording.edf_header)
            self._log_transforms = dict(recording.log_transforms)
            self._edf_annotations = recording.annotations_from_file
            self._edf_runs = recording.runs
            self._edf_record_duration = recording.edf_header.record_duration
            return recording

        self._is_discontinuous = False
        self._edf_annotations = []
        self._edf_runs = []

        # Only the SpikeInterface path needs pyedflib; the EDF+D reader is pure Python and NumPy, so
        # discontinuous files also convert where pyedflib is unavailable (e.g. M1 macs).
        get_package(
            package_name="pyedflib",
            excluded_platforms_and_python_versions=dict(darwin=dict(arm=["3.9"])),
        )

        self.extractor_kwargs = interface_kwargs.copy()
        self.extractor_kwargs.pop("verbose", None)
        self.extractor_kwargs.pop("es_key", None)
        self.extractor_kwargs.pop("metadata_key", None)
        self.extractor_kwargs.pop("channels_to_skip")
        self.extractor_kwargs["all_annotations"] = True
        self.extractor_kwargs["use_names_as_ids"] = True

        from ._edf_log_transform import (
            decode_log_transformed_signals,
            filtered_accounts_for_the_whole_unit_mix,
            read_log_transforms,
            read_signal_fields,
        )

        # Read the transforms first, so the reader below can be told to expect them. A file with none —
        # the overwhelmingly common case — takes exactly the path it always did.
        channels_to_skip = interface_kwargs.get("channels_to_skip")
        transforms = read_log_transforms(file_path=interface_kwargs["file_path"], channels_to_skip=channels_to_skip)

        # SpikeInterface sees the "Filtered" dimension as a non-voltage unit and warns about a mix. That
        # warning is unactionable for a transformed signal and, once decoded, untrue — but it is also the
        # only thing telling a user about a genuinely non-voltage channel, so suppress it only when
        # "Filtered" accounts for the whole mix.
        suppress_unit_mix_warning = False
        if transforms:
            _, dimensions, _ = read_signal_fields(file_path=interface_kwargs["file_path"])
            suppress_unit_mix_warning = filtered_accounts_for_the_whole_unit_mix(dimensions=dimensions)

        extractor_class = self.get_extractor_class()
        if suppress_unit_mix_warning:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Found a mix of voltage and non-voltage units.*",
                    category=UserWarning,
                )
                extractor_instance = extractor_class(**self.extractor_kwargs)
        else:
            # catch_warnings is not entered at all here: doing so resets filter state, which can make an
            # unrelated once-only warning print a second time.
            extractor_instance = extractor_class(**self.extractor_kwargs)

        # Captured before wrapping: the decoding recording does not forward neo_reader, and this is the
        # only place the unwrapped reader is in hand.
        self._edf_header = extractor_instance.neo_reader.edf_header

        extractor_instance, self._log_transforms = decode_log_transformed_signals(
            recording=extractor_instance, transforms=transforms
        )
        return extractor_instance

    def __init__(
        self,
        file_path: FilePath,
        *,
        verbose: bool = False,
        es_key: str | None = None,
        metadata_key: str | None = None,
        channels_to_skip: list | None = None,
        stream_name: str | None = None,
    ):
        """
        Load and prepare data for EDF.

        Plain EDF, continuous EDF+ (``EDF+C``) and discontinuous EDF+ (``EDF+D``) files are supported.

        Parameters
        ----------
        file_path : str or Path
            Path to the edf file
        verbose : bool, default: False
            Allows verbose.
        es_key : str, default: "ElectricalSeries"
            Key for the ElectricalSeries metadata
        metadata_key : str, optional
            Key that indexes this interface's entries in the dict-based metadata. Defaults to
            ``"edf_recording"``.
        channels_to_skip : list, default: None
            Channels to skip when adding the data to the nwbfile. These parameter can be used to skip non-neural
            channels that are present in the EDF file.
        stream_name : str, optional
            Name of the stream to read, as returned by ``get_stream_names``. A file that sampled some
            of its signals at a different rate than the rest carries more than one stream and cannot
            be read without naming one, since a single recording holds a single sampling rate. Not
            accepted for ``EDF+D`` files; use ``channels_to_skip`` there.

        """
        super().__init__(
            file_path=file_path,
            verbose=verbose,
            es_key=es_key,
            metadata_key=metadata_key,
            channels_to_skip=channels_to_skip,
            stream_name=stream_name,
        )

        if metadata_key is None:
            self.metadata_key = "edf_recording"

        self.edf_header = self._edf_header

        # On the EDF+D path channels_to_skip was already honored while reading.
        if channels_to_skip and not self._is_discontinuous:
            self.recording_extractor = self.recording_extractor.remove_channels(remove_channel_ids=channels_to_skip)

    def extract_nwb_file_metadata(self) -> dict:
        # The header names a single technician, while experimenter is a list of names.
        technician = self.edf_header["technician"]
        nwbfile_metadata = dict(
            session_start_time=self.edf_header["startdate"],
            experimenter=[technician] if technician else None,
        )

        # Filter empty values
        nwbfile_metadata = {property: value for property, value in nwbfile_metadata.items() if value}

        return nwbfile_metadata

    def extract_subject_metadata(self) -> dict:
        # "sex" is the current pyedflib key; "gender" is the older one, still populated, so fall back to
        # it rather than defaulting sex when the header in fact states it.
        sex = self.edf_header.get("sex") or self.edf_header.get("gender")
        subject_metadata = dict(
            subject_id=self.edf_header["patientcode"],
            date_of_birth=_parse_birthdate(self.edf_header["birthdate"]),
            sex=_parse_sex(sex),
        )

        # Filter empty values
        subject_metadata = {property: value for property, value in subject_metadata.items() if value}

        return subject_metadata

    @property
    def is_discontinuous(self) -> bool:
        """Whether the source file is marked as a discontinuous EDF+ (``EDF+D``) file."""
        return self._is_discontinuous

    @property
    def number_of_runs(self) -> int:
        """
        Number of contiguous runs of data records in the file.

        Always 1 for plain EDF and ``EDF+C``. For ``EDF+D`` this is the number of gap-separated runs
        actually found, which is frequently also 1 — exporters commonly label continuous recordings
        ``EDF+D``.
        """
        return self._number_of_segments

    @property
    def _time_shifts(self) -> list[float]:
        """
        Per-run seconds between the file's own timeline and the timeline being written.

        The record onsets and annotation timestamps parsed out of the file are relative to the start of
        the file, and alignment moves the recording without touching those parsed values, so anything
        derived from them has to be shifted or it points at the wrong moment in the NWB file.

        One shift per run rather than one for the whole file, because
        ``set_aligned_segment_starting_times`` moves each segment independently — and that is the natural
        API here, runs sitting at different times being the entire premise of ``EDF+D``. A single delta
        taken from segment 0 leaves every later run early by however much its own segment moved.
        """
        runs = getattr(self, "_edf_runs", None)
        if not runs:
            return []
        return [
            float(self.recording_extractor.get_start_time(segment_index=index)) - float(run.start_time)
            for index, run in enumerate(runs)
        ]

    def _shift_for_time(self, onset: float) -> float:
        """
        The shift applying to an annotation at ``onset`` on the file's timeline.

        An annotation belongs to the run whose span contains it. One falling in a gap has no run of its
        own, so it takes the preceding run's shift — with per-segment alignment a gap has no defined
        mapping, and staying with the run the annotation follows keeps it in order.
        """
        runs = getattr(self, "_edf_runs", None)
        shifts = self._time_shifts
        if not runs:
            return 0.0

        shift = shifts[0]
        for index, run in enumerate(runs):
            if onset < run.start_time:
                break
            shift = shifts[index]
            if onset < run.start_time + run.number_of_records * self._edf_record_duration:
                break
        return shift

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
        Add the EDF data to an NWBFile.

        Plain EDF, ``EDF+C``, and ``EDF+D`` files whose records turn out to be contiguous are written as
        a single ``ElectricalSeries``. A genuinely discontinuous ``EDF+D`` file is written as one
        ``ElectricalSeries`` per contiguous run, each placed at its true start time on the session
        timeline and sharing a single electrodes table.

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
            Whether to store raw integer samples with the gain and offset alongside, or float physical
            values. ``'physical_units'`` is the one that holds channels with heterogeneous offsets in a
            single series. See ``BaseRecordingExtractorInterface.add_to_nwbfile``.
        write_electrical_series : bool, default: True
            If False, only device, electrode groups and electrodes are written.
        iterator_type : {'v2', None}, default: 'v2'
            The type of iterator used for chunked data writing.
        iterator_options : dict, optional
            Options controlling the iterative write.
        always_write_timestamps : bool, default: False
            If True, always write an explicit timestamps array rather than a starting time and rate.
        write_annotations : bool, default: True
            If True and the file is ``EDF+D``, the annotations carried in the ``EDF Annotations`` signals
            are written to a ``TimeIntervals`` table named ``"annotations"``, with a ``source_file``
            column. If another EDF interface has already written that table to the file, the rows are
            added to it. Has no effect on plain EDF or ``EDF+C`` files, whose handling is unchanged.
        write_runs : bool, default: True
            If True and the file is discontinuous with more than one contiguous run, the run boundaries
            are written to a ``TimeIntervals`` table named ``"runs"``, or added to it as for the
            annotations.
        """
        # The full signature is redeclared rather than taking **conversion_options, because
        # get_conversion_options_schema is built from it: a catch-all erases every inherited option from
        # the schema and flips additionalProperties to True, so typos would validate.
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

        # Both tables come from neuroconv's own reader, which is used only for discontinuous files.
        # Files on the SpikeInterface path are untouched.
        if not self._is_discontinuous:
            return

        if write_annotations and getattr(self, "_edf_annotations", None):
            self._add_annotations_to_nwbfile(nwbfile=nwbfile)
        # A single run is indistinguishable from a continuous recording, so there is nothing to record.
        if write_runs and len(getattr(self, "_edf_runs", [])) > 1:
            self._add_runs_to_nwbfile(nwbfile=nwbfile)

    def _add_rows_to_table(self, nwbfile: NWBFile, name: str, description: str, columns: dict, rows: list[dict]):
        """
        Add ``rows`` to the ``TimeIntervals`` table ``name``, creating the table if the file has none.

        Several EDF files written into one NWB file through a ``ConverterPipe`` share the table, with a
        ``source_file`` column saying which file each row came from, so no file's rows are lost. The
        sub-interfaces from ``split_by_offset`` all come from one file and carry the same rows, so when
        every row is already in the table under this file's name, none is written again. A table of that
        name with other columns came from something else and is left alone, with a warning.
        """
        from pynwb.epoch import TimeIntervals

        source_file = Path(self.source_data["file_path"]).name
        columns = dict(columns, source_file="Name of the EDF file the row was read from.")
        rows = [dict(row, source_file=source_file) for row in rows]

        if nwbfile.intervals is None or name not in nwbfile.intervals:
            table = TimeIntervals(name=name, description=description)
            for column_name, column_description in columns.items():
                table.add_column(name=column_name, description=column_description)
            nwbfile.add_time_intervals(table)
        else:
            table = nwbfile.intervals[name]
            expected = {"start_time", "stop_time", *columns}
            if set(table.colnames) != expected:
                warnings.warn(
                    f"The NWB file already has an intervals table named {name!r}, with columns "
                    f"{sorted(table.colnames)} rather than an EDF {name} table's {sorted(expected)}. "
                    f"The {name} of '{source_file}' are not written.",
                    UserWarning,
                    stacklevel=4,
                )
                return
            # Whole rows are compared, source_file among them, rather than the name alone, so a second
            # file that merely shares this one's name — the same name in another directory — still has
            # its rows written.
            column_names = ["start_time", "stop_time", *columns]
            existing = set(zip(*(table[column_name][:] for column_name in column_names)))
            if {tuple(row[column_name] for column_name in column_names) for row in rows} <= existing:
                return

        for row in rows:
            table.add_row(**row)

    def _add_annotations_to_nwbfile(self, nwbfile: NWBFile):
        """Write the EDF+ annotations to a TimeIntervals table, on the session timeline."""
        rows = []
        for annotation in self._edf_annotations:
            duration = annotation["duration"] or 0.0
            onset = float(annotation["onset"])
            start_time = onset + self._shift_for_time(onset)
            rows.append(dict(start_time=start_time, stop_time=start_time + float(duration), label=annotation["text"]))
        self._add_rows_to_table(
            nwbfile=nwbfile,
            name="annotations",
            description="Annotations imported from the 'EDF Annotations' signals of EDF+ files.",
            columns={"label": "The EDF+ annotation text."},
            rows=rows,
        )

    def _add_runs_to_nwbfile(self, nwbfile: NWBFile):
        """
        Write the boundaries of each contiguous run of data records to a TimeIntervals table.

        A dedicated table is used rather than ``nwbfile.epochs``, for two reasons: ``epochs`` is a
        shared singleton that another interface in the same ``ConverterPipe`` — or the caller of a
        pre-built ``NWBFile`` — may already have populated, and ``add_epoch_column`` cannot extend a
        table that has rows, so reaching into it raises
        ``ValueError: column must have the same number of rows as 'id'``. This also keeps the runs
        consistent with how the annotations are written.
        """
        time_shifts = self._time_shifts
        rows = []
        for run_index, run in enumerate(self._edf_runs):
            start_time = float(run.start_time) + time_shifts[run_index]
            rows.append(
                dict(
                    start_time=start_time,
                    stop_time=start_time + run.number_of_records * self._edf_record_duration,
                    run_index=run_index,
                )
            )
        self._add_rows_to_table(
            nwbfile=nwbfile,
            name="runs",
            description=(
                "Contiguous runs of data records in discontinuous EDF+ (EDF+D) files. Each run is "
                "written as its own ElectricalSeries, positioned at the run's start time."
            ),
            columns={"run_index": "0-based index of the contiguous run of data records within its EDF file."},
            rows=rows,
        )

    @property
    def log_transformed_channels(self) -> dict:
        """
        Channel label to logarithmic transform, for every signal carrying transformed data.

        Empty for an ordinary file. Such a signal stores logarithms rather than measurements, so its
        samples are decoded while reading and the recording holds physical values as floats.
        """
        return dict(self._log_transforms)

    def get_metadata(self, *, use_new_metadata_format: bool = True) -> DeepDict:
        metadata = super().get_metadata(use_new_metadata_format=use_new_metadata_format)
        nwbfile_metadata = self.extract_nwb_file_metadata()
        metadata["NWBFile"].update(nwbfile_metadata)

        subject_metadata = self.extract_subject_metadata()
        if subject_metadata:
            # NOTE: assigning rather than updating in place. ``metadata`` is a DeepDict (a defaultdict),
            # and ``dict.get`` does not go through ``__missing__``, so the previous
            # ``metadata.get("Subject", dict()).update(...)`` mutated a throwaway dict and every EDF
            # file silently lost its subject metadata.
            # Once "Subject" is present the metadata schema requires subject_id, species and sex, so
            # fill them whenever the header carried anything at all; otherwise reading a patient code
            # out of the file would turn valid metadata into invalid metadata. These are fallbacks only:
            # extract_subject_metadata supplies sex when the header states it.
            subject_metadata.setdefault("subject_id", "Unknown")
            # "Unknown species" is shaped like a binomial name on purpose: nwbinspector's
            # check_subject_species_form requires "Genus species" or an NCBI taxonomy URL, so a plainer
            # "unknown" would be flagged on every converted file.
            subject_metadata.setdefault("species", "Unknown species")
            subject_metadata.setdefault("sex", "U")
            # NOTE: the ``.get`` here is a read, not the bug fixed above — it deliberately avoids
            # materialising "Subject" on this defaultdict, and its result is used rather than mutated.
            metadata["Subject"] = {**metadata.get("Subject", dict()), **subject_metadata}

        return metadata
