"""
Pure-Python reader for discontinuous EDF+ (EDF+D) files.

EDF+D is the one part of EDF+ that is not backwards compatible with plain EDF: its data records are
not required to be contiguous in time, so a record's start time cannot be inferred from its index.
Instead the format carries the start time of every data record inside the mandatory
``EDF Annotations`` signal — the first TAL (Time-stamped Annotations List) of each record is a
"time-keeping" annotation whose onset is the number of seconds between the file start and that
record. See https://www.edfplus.info/specs/edfplus.html.

Neither of the readers neuroconv would normally use can read these files:

* ``pyedflib`` (i.e. the EDFlib C library) refuses them outright with
  ``"the file is discontinuous and cannot be read"``.
* neo's ``EDFRawIO`` rejects them in ``_parse_header`` and models every file as a single contiguous
  segment (``nb_segment = [1]``, ``t_stop = record_duration * n_records``), so it has no way to
  represent per-record start times even if the check were removed.

So this module parses the container directly. Samples are read lazily: EDF data records are
fixed-size, so the byte offset of any sample is pure arithmetic and only the records overlapping a
requested chunk are ever read from disk. That keeps memory flat for the ~1 GB clinical recordings
this format shows up in.
"""

import re
import warnings
from datetime import datetime
from typing import NamedTuple

import numpy as np
from pydantic import FilePath

# The logarithmic transform is shared with the SpikeInterface path, which uses it for continuous files.
# One definition, so the two decode paths cannot drift.
from ._edf_log_transform import (
    _LOG_TRANSFORM_DIMENSION,
    _parse_log_transform,
    _unit_to_microvolts,
    _warn_assumed_microvolts,
)

# The label EDF+ reserves for the annotations signal.
_ANNOTATIONS_SIGNAL_LABEL = "EDF Annotations"

# TAL separators, defined by the spec as single bytes with these values.
_TAL_ONSET_DURATION_SEPARATOR = b"\x15"  # byte 21, between onset and duration
_TAL_TEXT_SEPARATOR = b"\x14"  # byte 20, after the timestamp and between annotation texts
_TAL_TERMINATOR = b"\x00"  # byte 0, ends a TAL

# Fixed sizes, in bytes, of the static part of the EDF header and of each per-signal field.
_STATIC_HEADER_SIZE = 256
_SAMPLE_SIZE = 2  # EDF stores 2-byte little-endian two's-complement integers

_MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]


def _decode_field(raw: bytes) -> str:
    """
    Decode one ASCII header field.

    EDF specifies that header fields are ASCII, left-justified and space-padded, but real files pad
    with NUL bytes instead — which is exactly what makes EDFlib reject some otherwise-valid files
    (``"not EDF(+) or BDF(+) compliant (Number of Datarecords)"``). Treat NUL as padding.

    Decoded as latin-1 rather than ASCII: every byte maps to a character, so nothing is ever lost, and
    the micro sign that writers put in a physical dimension — byte 0xB5, outside ASCII — arrives as "µ"
    instead of a replacement character. ASCII files decode identically either way.
    """
    return raw.decode("latin-1").replace("\x00", " ").strip()


def _decode_int(raw: bytes, field_name: str) -> int:
    """Decode an integer header field, naming the field if it cannot be read."""
    return int(_decode_float(raw=raw, field_name=field_name))


def _decode_float(raw: bytes, field_name: str) -> float:
    """
    Decode a numeric header field, naming the field if it cannot be read.

    This reader exists for files other EDF readers refuse, so its error message is often the only
    diagnostic a user gets; a bare ``could not convert string to float: ''`` does not identify which
    field of which signal is malformed, or that the header was truncated.
    """
    text = _decode_field(raw)
    try:
        return float(text)
    except ValueError:
        raise ValueError(
            f"Could not read the EDF header field {field_name!r}: expected a number, got {text!r}. "
            "The file may be truncated or malformed."
        ) from None


def _parse_start_datetime(startdate: str, starttime: str, recording_field: str) -> datetime | None:
    """
    Recover the recording start datetime from the EDF header.

    The header's own ``startdate`` field only has a two-digit year, so EDF+ additionally puts a
    four-digit ``Startdate DD-MMM-YYYY`` at the start of the local recording identification field.
    Prefer that; fall back to the two-digit field using the clipping rule from the EDF spec
    (85-99 mean 1985-1999, 00-84 mean 2000-2084).
    """
    hour, minute, second = None, None, None
    time_match = re.match(r"^(\d{1,2})[.:](\d{1,2})[.:](\d{1,2})", starttime.strip())
    if time_match:
        hour, minute, second = (int(group) for group in time_match.groups())

    year, month, day = None, None, None
    recording_match = re.match(r"^Startdate\s+(\d{1,2})-([A-Za-z]{3})-(\d{4})", recording_field.strip())
    if recording_match:
        day = int(recording_match.group(1))
        month_name = recording_match.group(2).upper()
        if month_name in _MONTHS:
            month = _MONTHS.index(month_name) + 1
        year = int(recording_match.group(3))

    def assemble(year, month, day) -> datetime | None:
        if None in (year, month, day) or hour is None:
            return None
        try:
            return datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=second)
        except ValueError:
            return None

    # Fall back to the header's own startdate whenever the recording field did not yield a *usable*
    # date — not merely when its month name was unrecognized. De-identification that zeroes the day
    # ("Startdate 00-JAN-2021") parses but cannot be assembled, and the header field is often intact.
    from_recording_field = assemble(year, month, day)
    if from_recording_field is not None:
        return from_recording_field

    date_match = re.match(r"^(\d{1,2})[.](\d{1,2})[.](\d{1,2})$", startdate.strip())
    if date_match:
        day, month, two_digit_year = (int(group) for group in date_match.groups())
        year = 1900 + two_digit_year if two_digit_year >= 85 else 2000 + two_digit_year
        return assemble(year, month, day)
    return None


class EDFHeader:
    """The parsed EDF/EDF+ header, plus the derived record geometry."""

    def __init__(self, fields: dict):
        self.__dict__.update(fields)

    @property
    def is_discontinuous(self) -> bool:
        return self.reserved.upper().startswith("EDF+D")

    @property
    def is_edf_plus(self) -> bool:
        return self.reserved.upper().startswith("EDF+")

    @property
    def annotations_signal_indices(self) -> list[int]:
        """
        Indices of every ``EDF Annotations`` signal, in file order.

        The spec permits more than one: "Additional 'EDF Annotations' signals may be defined according
        to the same specification". Writers use them when the annotations of a long recording do not
        fit in a single signal, so all of them must be kept out of the data channels.
        """
        return [index for index, label in enumerate(self.labels) if label == _ANNOTATIONS_SIGNAL_LABEL]

    @property
    def annotations_signal_index(self) -> int | None:
        """Index of the first ``EDF Annotations`` signal — the one carrying the time-keeping TALs."""
        indices = self.annotations_signal_indices
        return indices[0] if indices else None

    @property
    def samples_per_record_total(self) -> int:
        """Number of 2-byte samples in one data record, across all signals."""
        return int(sum(self.samples_per_record))

    @property
    def record_size_bytes(self) -> int:
        return self.samples_per_record_total * _SAMPLE_SIZE

    def sample_offset_in_record(self, signal_index: int) -> int:
        """Index (in samples, not bytes) at which a signal's block starts within a data record."""
        return int(sum(self.samples_per_record[:signal_index]))


def read_edf_header(file) -> EDFHeader:
    """
    Parse the EDF header from an open binary file object.

    Parameters
    ----------
    file : file-like
        A file object opened in binary mode, positioned anywhere (it is rewound).

    Returns
    -------
    EDFHeader
        The parsed header. ``number_of_records`` is always the number of records the file actually
        contains, which is not necessarily the number its header claims.
    """
    file.seek(0)
    fields = dict(
        version=_decode_field(file.read(8)),
        patient=_decode_field(file.read(80)),
        recording=_decode_field(file.read(80)),
        startdate=_decode_field(file.read(8)),
        starttime=_decode_field(file.read(8)),
        header_size_bytes=_decode_int(file.read(8), "number of bytes in header record"),
        reserved=_decode_field(file.read(44)),
        number_of_records=_decode_int(file.read(8), "number of data records"),
        record_duration=_decode_float(file.read(8), "duration of a data record"),
    )
    number_of_signals = _decode_int(file.read(4), "number of signals")
    fields["number_of_signals"] = number_of_signals
    if number_of_signals <= 0:
        raise ValueError(f"The EDF header declares {number_of_signals} signals; the file cannot be read.")

    fields["labels"] = [_decode_field(file.read(16)) for _ in range(number_of_signals)]
    fields["transducers"] = [_decode_field(file.read(80)) for _ in range(number_of_signals)]
    fields["dimensions"] = [_decode_field(file.read(8)) for _ in range(number_of_signals)]
    fields["physical_min"] = [
        _decode_float(file.read(8), f"physical minimum of signal {index}") for index in range(number_of_signals)
    ]
    fields["physical_max"] = [
        _decode_float(file.read(8), f"physical maximum of signal {index}") for index in range(number_of_signals)
    ]
    fields["digital_min"] = [
        _decode_int(file.read(8), f"digital minimum of signal {index}") for index in range(number_of_signals)
    ]
    fields["digital_max"] = [
        _decode_int(file.read(8), f"digital maximum of signal {index}") for index in range(number_of_signals)
    ]
    fields["prefilters"] = [_decode_field(file.read(80)) for _ in range(number_of_signals)]
    fields["samples_per_record"] = [
        _decode_int(file.read(8), f"number of samples per record of signal {index}")
        for index in range(number_of_signals)
    ]

    fields["start_datetime"] = _parse_start_datetime(
        startdate=fields["startdate"], starttime=fields["starttime"], recording_field=fields["recording"]
    )

    header = EDFHeader(fields)

    # The spec fixes the header size at 256 bytes plus 256 per signal. A value that disagrees means the
    # geometry is wrong, and trusting it would silently read header padding as samples.
    expected_header_size = _STATIC_HEADER_SIZE * (number_of_signals + 1)
    if header.header_size_bytes != expected_header_size:
        raise ValueError(
            f"The EDF header declares a header size of {header.header_size_bytes} bytes, but a file with "
            f"{number_of_signals} signals must have {expected_header_size}. The file is malformed."
        )

    # Every signal declaring zero samples per record makes a data record zero bytes long, and the record
    # arithmetic below divides by that. This runs for every EDF file now, because the routing decision
    # parses the header before SpikeInterface is consulted, so a bare ZeroDivisionError here would be the
    # first thing a user of a degenerate plain EDF saw.
    if header.record_size_bytes == 0:
        raise ValueError(
            "Every signal in this EDF file declares zero samples per data record, so the file contains no "
            "samples and its record geometry is undefined. The header is malformed."
        )

    # Always derive the record count from the file size rather than only when the header says -1
    # (which it is allowed to while a file is being written). An interrupted acquisition leaves a
    # header claiming more records than were written, and trusting it defers the failure to read time —
    # potentially hours into a conversion — as an opaque reshape error.
    file.seek(0, 2)
    available_records = max(file.tell() - header.header_size_bytes, 0) // header.record_size_bytes
    if header.number_of_records > available_records:
        # Say so rather than truncating quietly: the header promised data the file does not contain, which
        # means the acquisition was interrupted, and silently losing records is its own defect.
        warnings.warn(
            f"This EDF file's header declares {header.number_of_records} data records but only "
            f"{available_records} are present, so the recording is {header.number_of_records - available_records} "
            "record(s) shorter than the header claims. The acquisition was probably interrupted.",
            UserWarning,
            stacklevel=3,
        )
        header.number_of_records = available_records
    elif header.number_of_records < 0:
        header.number_of_records = available_records
    if header.number_of_records == 0:
        raise ValueError("The EDF file contains no complete data records.")

    return header


def needs_neuroconv_reader(file_path: FilePath) -> tuple[bool, bool]:
    """
    Decide whether a file has to be read by neuroconv's own reader rather than SpikeInterface's.

    Two properties disqualify SpikeInterface's reader, for different reasons:

    * discontinuity (``EDF+D``), which it refuses outright and cannot represent, and
    * a logarithmically transformed signal, which it reads *successfully* and silently wrong, applying
      the header's linear gain and offset to what are actually logarithms. Values come back off by
      many orders of magnitude with no error and no warning.

    Parameters
    ----------
    file_path : FilePath
        Path to the ``.edf`` file.

    Returns
    -------
    tuple of bool
        ``(is_discontinuous, has_log_transformed_signal)``.
    """
    with open(file_path, "rb") as file:
        header = read_edf_header(file)
    has_log_transform = any(dimension.strip().upper() == _LOG_TRANSFORM_DIMENSION for dimension in header.dimensions)
    return header.is_discontinuous, has_log_transform


def is_discontinuous_edf(file_path: FilePath) -> bool:
    """
    Return True if the file's header marks it as discontinuous EDF+ (``EDF+D``).

    Reads only the 44-byte reserved field, so this is cheap enough to call before deciding which
    reader to use.

    Parameters
    ----------
    file_path : FilePath
        Path to the ``.edf`` file.

    Returns
    -------
    bool
        True if the header's reserved field starts with ``EDF+D``.
    """
    with open(file_path, "rb") as file:
        file.seek(192)
        return _decode_field(file.read(44)).upper().startswith("EDF+D")


def _is_onset_field(field: bytes) -> bool:
    """
    True if ``field`` is an EDF+ onset — a decimal number carrying a mandatory explicit sign.

    The sign is what makes this usable to find TAL boundaries: an annotation text such as ``601`` is
    not a candidate, while ``+1.000000`` is. See :func:`_split_into_tals`.
    """
    if field[:1] not in (b"+", b"-"):
        return False
    try:
        float(field.partition(_TAL_ONSET_DURATION_SEPARATOR)[0].decode("ascii", errors="replace"))
    except ValueError:
        return False
    return True


def _split_into_tals(chunk: bytes) -> list[list[bytes]]:
    """
    Split one NUL-delimited chunk into TALs, each returned as ``[timestamp, *annotation_fields]``.

    A conformant writer ends every TAL with a NUL byte, so a chunk holds exactly one TAL and this
    returns it unchanged. Some writers only NUL-pad the tail of the block and never terminate a TAL,
    packing a record's time-keeping annotation and its real annotations into one NUL-free run of
    bytes; Nihon Kohden's ``EDF+D`` export does this. Splitting those apart here is what keeps the
    time-keeping onset — and therefore the whole file — readable.

    Two structural facts make the split unambiguous for the shapes seen in practice. The time-keeping
    annotation carries exactly one annotation text and that text is empty, so any field following
    that empty one begins a new TAL. And an EDF+ onset always carries an explicit sign, so a signed
    number appearing where an annotation text is expected begins one too.
    """
    fields = chunk.split(_TAL_TEXT_SEPARATOR)
    tals = []
    current = [fields[0]]
    for field in fields[1:]:
        # `len(current) > 1` means the TAL already has its timestamp and at least one text, so a
        # signed number here cannot be this TAL's onset and must start the next one.
        starts_new_tal = len(current) > 1 and _is_onset_field(field)
        # The time-keeping annotation is `onset[20][20]` and owns nothing beyond that empty text.
        completes_time_keeping = len(current) == 2 and not current[1] and field
        if starts_new_tal or completes_time_keeping:
            tals.append(current)
            current = [field]
        else:
            current.append(field)
    tals.append(current)
    return tals


def _parse_tals(raw: bytes) -> list[tuple[float, float | None, list[str]]]:
    """
    Parse the TALs in one record's ``EDF Annotations`` block.

    Each TAL is ``Onset[21]Duration[20]Text[20]...[0]``, where ``[21]`` and the duration are optional,
    and the terminating ``[0]`` is omitted by some writers (see :func:`_split_into_tals`). Returns
    ``(onset, duration, texts)`` per TAL, skipping anything unparsable — trailing NUL padding is
    normal and must not be treated as an error.
    """
    tals = []
    for chunk in raw.split(_TAL_TERMINATOR):
        if not chunk.strip():
            continue
        for fields in _split_into_tals(chunk):
            onset_field, _, duration_field = fields[0].partition(_TAL_ONSET_DURATION_SEPARATOR)
            try:
                onset = float(onset_field.decode("ascii", errors="replace"))
            except ValueError:
                continue
            duration = None
            if duration_field.strip():
                try:
                    duration = float(duration_field.decode("ascii", errors="replace"))
                except ValueError:
                    duration = None

            texts = [field.decode("utf-8", errors="replace").strip() for field in fields[1:]]
            tals.append((onset, duration, [text for text in texts if text]))
    return tals


def _consecutive_blocks(indices: list[int]) -> list[list[int]]:
    """Group an ascending list of record indices into maximal runs of consecutive values."""
    blocks = []
    for index in indices:
        if blocks and index == blocks[-1][-1] + 1:
            blocks[-1].append(index)
        else:
            blocks.append([index])
    return blocks


def _place_records_between_known_onsets(
    *, onsets: np.ndarray, missing: list[int], record_duration: float, number_of_records: int
) -> list[int]:
    """
    Recover the onsets of records whose time-keeping annotation was unreadable, wherever the records
    on either side prove that nothing is missing between them.

    The position is *derived*, not assumed. If the known onsets bracketing a block of unreadable ones
    span exactly the number of records in between, then no gap can be hiding inside that block and
    every onset in it follows by arithmetic. A block that fails this test, or that runs to either end
    of the file where there is no anchor on one side, is left alone and reported to the caller —
    ``EDF+D`` exists precisely to say that contiguity cannot be taken for granted, so a record whose
    neighbours do not vouch for it stays unplaced.

    Onsets are written in place. Returns the record indices still unplaced, in ascending order.
    """
    still_missing = []
    for block in _consecutive_blocks(missing):
        before, after = block[0] - 1, block[-1] + 1
        if before < 0 or after >= number_of_records:
            still_missing.extend(block)
            continue
        # Half a record duration is far tighter than any gap a writer would bother to record, and
        # loose enough to absorb the rounding in onsets printed as fixed-width decimals.
        expected_span = (len(block) + 1) * record_duration
        if abs((onsets[after] - onsets[before]) - expected_span) > 0.5 * record_duration:
            still_missing.extend(block)
            continue
        for position, record_index in enumerate(block, start=1):
            onsets[record_index] = onsets[before] + position * record_duration
    return still_missing


def read_record_onsets_and_annotations(file, header: EDFHeader) -> tuple[np.ndarray, list[dict]]:
    """
    Read every data record's start time, plus all real annotations, from the annotations signals.

    The first TAL of the first ``EDF Annotations`` signal in each record is the mandatory time-keeping
    annotation: its onset is the record's start time relative to the file start, and its annotation
    text is empty — which is what distinguishes it from a real annotation that merely happens to come
    first. Every other TAL, in that signal or any additional annotations signal, is a real annotation
    timestamped on the same file-relative timeline.

    Parameters
    ----------
    file : file-like
        A file object opened in binary mode.
    header : EDFHeader
        The parsed header of the same file.

    Returns
    -------
    tuple
        ``(onsets, annotations)``. ``onsets`` has one float per data record; ``annotations`` is a list
        of ``dict(onset=..., duration=..., text=...)``.

    Where a record's time-keeping onset cannot be read, the records either side of it are given the
    chance to prove that no gap hides there, in which case its onset follows from theirs and only a
    warning is issued. See :func:`_place_records_between_known_onsets`.

    Raises
    ------
    ValueError
        If the file is ``EDF+D`` and any record's time-keeping onset can neither be read nor derived
        from its neighbours — the record's position on the timeline is then genuinely unknown, and
        assuming contiguity would silently misplace data, which is the one thing ``EDF+D`` exists to
        express.
    """
    annotations_indices = header.annotations_signal_indices
    if not annotations_indices:
        raise ValueError(
            "This EDF+ file is marked discontinuous (EDF+D) but has no 'EDF Annotations' signal, so the "
            "start time of each data record cannot be recovered. The file does not conform to EDF+."
        )
    time_keeping_index = annotations_indices[0]

    layouts = {
        index: (
            header.sample_offset_in_record(index) * _SAMPLE_SIZE,
            header.samples_per_record[index] * _SAMPLE_SIZE,
        )
        for index in annotations_indices
    }

    onsets = np.empty(header.number_of_records, dtype="float64")
    annotations = []
    records_missing_onset = []
    for record_index in range(header.number_of_records):
        record_start = header.header_size_bytes + record_index * header.record_size_bytes
        for signal_index in annotations_indices:
            byte_offset, block_size = layouts[signal_index]
            file.seek(record_start + byte_offset)
            tals = _parse_tals(file.read(block_size))

            if signal_index == time_keeping_index:
                # Only an *empty-text* leading TAL is the time-keeping annotation. Trusting tals[0]
                # unconditionally would silently adopt a real annotation's onset as the record start.
                if tals and not tals[0][2]:
                    onsets[record_index] = tals[0][0]
                    tals = tals[1:]
                else:
                    onsets[record_index] = record_index * header.record_duration
                    records_missing_onset.append(record_index)

            for onset, duration, texts in tals:
                for text in texts:
                    annotations.append(dict(onset=onset, duration=duration, text=text))

    if records_missing_onset and header.is_discontinuous:
        unplaced = _place_records_between_known_onsets(
            onsets=onsets,
            missing=records_missing_onset,
            record_duration=header.record_duration,
            number_of_records=header.number_of_records,
        )
        recovered = len(records_missing_onset) - len(unplaced)
        if recovered:
            warnings.warn(
                f"{recovered} of {header.number_of_records} data records in this discontinuous EDF+ "
                "(EDF+D) file carry no readable time-keeping annotation, so the file does not conform "
                "to EDF+. The records on either side of each of them span exactly the intervening "
                "number of records, which rules out a gap there, so their start times follow from "
                "their neighbours and the recording timeline is unaffected.",
                UserWarning,
                stacklevel=2,
            )
        if unplaced:
            shown = unplaced[:5]
            byte_offset, block_size = layouts[time_keeping_index]
            file.seek(header.header_size_bytes + unplaced[0] * header.record_size_bytes + byte_offset)
            sample = file.read(block_size).rstrip(_TAL_TERMINATOR)
            raise ValueError(
                f"{len(unplaced)} of {header.number_of_records} data records in this discontinuous "
                f"EDF+ (EDF+D) file carry no readable time-keeping annotation, and their neighbours do "
                f"not rule out a gap across them, so their position on the recording timeline cannot be "
                f"recovered (records {shown}{' ...' if len(unplaced) > len(shown) else ''}). Assuming "
                "the records are contiguous would silently misplace every sample after the first gap, "
                "which is the one thing EDF+D exists to rule out.\n"
                f"For reference, the 'EDF Annotations' block of record {unplaced[0]} reads "
                f"{sample!r}. A conformant block opens with a signed onset, then byte 20, then an "
                "empty annotation text, then byte 20 — for example b'+1.234\\x14\\x14'. Departures "
                "seen in practice are a comma as the decimal separator, a missing sign, and a real "
                "annotation placed ahead of the time-keeping one."
            )

    return onsets, annotations


class EDFRun(NamedTuple):
    """One contiguous run of data records, and where it sits on the recording timeline."""

    first_record_index: int
    number_of_records: int
    start_time: float


def group_records_into_runs(onsets: np.ndarray, samples_per_record: int, sampling_frequency: float) -> list[EDFRun]:
    """
    Group consecutive data records into contiguous runs.

    A record continues the current run when it begins exactly where the previous record ended.
    Anything else — a forward gap or a backward jump — starts a new run.

    The comparison is made in whole samples rather than in seconds. Time-keeping onsets are ASCII
    decimals, so they carry rounding error whenever the record duration does not terminate in the
    printed number of decimals; rounding the inter-record distance to the nearest sample absorbs that
    error exactly, while still catching every real gap (a gap of less than one sample has no missing
    samples to represent). Comparing against a seconds tolerance instead would let accumulated printing
    error split a perfectly continuous recording into thousands of spurious runs.

    Parameters
    ----------
    onsets : numpy.ndarray
        One start time per data record, relative to the start of the file.
    samples_per_record : int
        Number of samples each data channel contributes per record.
    sampling_frequency : float
        Sampling rate shared by the data channels, in Hz.

    Returns
    -------
    list of EDFRun
        One entry per contiguous run, in file order.
    """
    if len(onsets) == 0:
        return []

    runs = []
    run_start_index = 0
    for index in range(1, len(onsets)):
        distance_in_samples = round((onsets[index] - onsets[index - 1]) * sampling_frequency)
        if distance_in_samples != samples_per_record:
            runs.append(
                EDFRun(
                    first_record_index=run_start_index,
                    number_of_records=index - run_start_index,
                    start_time=float(onsets[run_start_index]),
                )
            )
            run_start_index = index
    runs.append(
        EDFRun(
            first_record_index=run_start_index,
            number_of_records=len(onsets) - run_start_index,
            start_time=float(onsets[run_start_index]),
        )
    )
    return runs


def _unknown_to_empty(value: str) -> str:
    """EDF+ writes a bare ``X`` for an unknown subfield; treat it as absent."""
    return "" if value.strip() in ("", "X") else value.strip()


def edf_plus_header_fields(header: EDFHeader) -> dict:
    """
    Reshape a parsed EDF+ header into the same dict pyedflib's ``getHeader()`` returns.

    This lets the EDF interface pull metadata through one code path regardless of which reader
    produced the file. EDF+ packs subfields into the two free-text identification fields:
    ``patient`` is ``code sex birthdate name`` and ``recording`` is
    ``Startdate DD-MMM-YYYY admincode technician equipment``.

    Parameters
    ----------
    header : EDFHeader
        A parsed EDF header.

    Returns
    -------
    dict
        The same keys pyedflib's ``getHeader()`` provides, with unknown subfields as empty strings.
    """
    patient_parts = header.patient.split()
    recording_parts = header.recording.split()

    fields = dict(
        startdate=header.start_datetime,
        patientcode=_unknown_to_empty(patient_parts[0]) if len(patient_parts) > 0 else "",
        sex=_unknown_to_empty(patient_parts[1]) if len(patient_parts) > 1 else "",
        birthdate=_unknown_to_empty(patient_parts[2]) if len(patient_parts) > 2 else "",
        patientname=_unknown_to_empty(patient_parts[3]) if len(patient_parts) > 3 else "",
        admincode="",
        technician="",
        equipment="",
        recording_additional="",
        patient_additional="",
    )
    # recording_parts is ["Startdate", date, admincode, technician, equipment, ...]
    if len(recording_parts) > 2 and recording_parts[0].lower() == "startdate":
        for key, position in (("admincode", 2), ("technician", 3), ("equipment", 4)):
            if len(recording_parts) > position:
                fields[key] = _unknown_to_empty(recording_parts[position])
    return fields


def _build_channel_layout(header: EDFHeader, channels_to_skip: list | None) -> dict:
    """
    Work out which signals become data channels, and their rate, byte layout and scaling.

    Kept separate from the extractor class so it can be exercised without SpikeInterface installed.
    """
    # Every annotations signal is excluded, not just the time-keeping one: the spec permits additional
    # ones, and a second would otherwise become a bogus data channel carrying TAL bytes.
    annotations_indices = set(header.annotations_signal_indices)
    unknown_skips = sorted(set(channels_to_skip or []) - set(header.labels))
    if unknown_skips:
        raise ValueError(
            f"channels_to_skip names channels that are not in this file: {unknown_skips}. "
            f"Available channels: {sorted(set(header.labels))}."
        )

    data_signal_indices = [
        index
        for index in range(header.number_of_signals)
        if index not in annotations_indices and not (channels_to_skip and header.labels[index] in channels_to_skip)
    ]
    if not data_signal_indices:
        raise ValueError("No data channels remain after excluding the annotations signal and channels_to_skip.")

    # EDF does not require unique or non-empty labels, but they become the NWB channel ids, so reject
    # them up front rather than failing deep inside the electrodes-table write. The SpikeInterface path
    # fails fast here too ("use_names_as_ids=True is not possible, channel names are not unique").
    channel_names = [header.labels[index] for index in data_signal_indices]
    if any(not name for name in channel_names):
        raise ValueError("This EDF file has a data channel with a blank label, which cannot be used as a channel id.")
    duplicates = sorted({name for name in channel_names if channel_names.count(name) > 1})
    if duplicates:
        raise ValueError(
            f"This EDF file has duplicate channel labels {duplicates}, which cannot be used as channel ids. "
            "Use channels_to_skip to drop the duplicates."
        )

    sampling_frequencies = {header.samples_per_record[index] / header.record_duration for index in data_signal_indices}
    if len(sampling_frequencies) != 1:
        raise ValueError(
            "EDF channels have differing sampling rates "
            f"({sorted(sampling_frequencies)} Hz); only uniform-rate recordings are supported. "
            "Use channels_to_skip to drop off-rate channels."
        )

    # Logarithmically transformed signals are decoded at read time, so their gain is only the unit
    # scaling and their offset is zero: the exponential relationship between stored integer and physical
    # value cannot be expressed as the scalar gain+offset an NWB ElectricalSeries applies.
    log_transforms = {
        position: _parse_log_transform(dimension=header.dimensions[index], prefilter=header.prefilters[index])
        for position, index in enumerate(data_signal_indices)
    }
    log_transforms = {position: transform for position, transform in log_transforms.items() if transform}

    gains_to_microvolts = np.empty(len(data_signal_indices), dtype="float64")
    offsets_to_microvolts = np.empty(len(data_signal_indices), dtype="float64")
    assumed_microvolts: dict[str, list[str]] = {}
    for position, index in enumerate(data_signal_indices):
        if position in log_transforms:
            transform_unit = log_transforms[position].unit
            factor = _unit_to_microvolts(unit=transform_unit)
            if factor is None:
                assumed_microvolts.setdefault(transform_unit, []).append(header.labels[index])
                factor = 1.0
            gains_to_microvolts[position] = factor
            offsets_to_microvolts[position] = 0.0
            continue
        physical_range = header.physical_max[index] - header.physical_min[index]
        # NOTE: the digital span is deliberately the *number of codes* (max - min + 1) rather than the
        # distance between the extremes (max - min), which is what the spec's endpoint mapping and
        # pyedflib/EDFlib use. This matches neo's convention, and therefore the EDF/EDF+C path through
        # SpikeInterface, which matters more than the ~15 ppm of gain it costs:
        #
        #   for the near-universal symmetric digital range (-32768/32767) this collapses the offset to
        #   exactly 0, whereas dividing by (max - min) leaves half an LSB of DC that scales with each
        #   channel's physical range. Channels of different ranges — EEG in uV beside ECG, EOG or SpO2,
        #   i.e. the ordinary clinical layout — then get *different* offsets, and a single NWB
        #   ElectricalSeries carries only one scalar offset. The same recording would convert when
        #   labelled EDF+C and fail when labelled EDF+D.
        #
        # The half-LSB term is physically meaningless, so the consistency is worth more than the purity.
        digital_span = header.digital_max[index] - header.digital_min[index] + 1
        gain = physical_range / digital_span if digital_span else 1.0
        offset = header.physical_min[index] - header.digital_min[index] * gain
        to_microvolts = _unit_to_microvolts(unit=header.dimensions[index])
        if to_microvolts is None:
            # Assume microvolts rather than refusing the channel, and say so once per distinct unit.
            assumed_microvolts.setdefault(header.dimensions[index], []).append(header.labels[index])
            to_microvolts = 1.0
        gains_to_microvolts[position] = gain * to_microvolts
        offsets_to_microvolts[position] = offset * to_microvolts

    _warn_assumed_microvolts(unit_to_channel_names=assumed_microvolts)

    return dict(
        channel_names=channel_names,
        log_transforms=log_transforms,
        # float64 rather than float32: the transform's whole purpose is dynamic range, and the spec's own
        # accuracy table reaches 1.4e68, well past what float32 can hold.
        dtype="float64" if log_transforms else "int16",
        sampling_frequency=sampling_frequencies.pop(),
        samples_per_record=header.samples_per_record[data_signal_indices[0]],
        sample_offsets_in_record=np.array(
            [header.sample_offset_in_record(index) for index in data_signal_indices], dtype="int64"
        ),
        gains_to_microvolts=gains_to_microvolts,
        offsets_to_microvolts=offsets_to_microvolts,
    )
