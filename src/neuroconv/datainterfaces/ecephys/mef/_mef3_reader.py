"""
Reader for MEF 3.0 sessions (Multiscale Electrophysiology Format, Mayo Systems Electrophysiology Lab).

A session is a ``.mefd`` directory holding one ``.timd`` directory per time series channel, each holding one
``.segd`` directory per segment, each holding three files:

* ``.tmet``, the segment's metadata: a 16 kB file of fields at fixed offsets.
* ``.tidx``, the block index: one 56-byte entry per block, giving its position in the data file, its start
  time and its number of samples.
* ``.tdat``, the samples, as RED-compressed blocks (see :mod:`._red`).

Records (notes, seizure marks, stimulation events, ...) may sit beside them at any of the three levels, as a
``.ridx`` index and ``.rdat`` data file.

Every offset here is meflib's (``meflib.h`` and ``mefrec.h`` in pymef 1.4.8, the reference implementation),
not the MEF 3.0 specification's prose. pymef itself is not used: it is a test oracle only, and ships wheels
for x86_64 alone.

Only what is needed to place the samples on the timeline and scale them is read. In particular the subject
name and ID fields of section 3 are never read, so they cannot reach the NWB file.

Times in MEF are microseconds since the Unix epoch (µUTC). A writer may store them offset by the session's
recording time offset, marking such a time by making it negative; it is undone here as meflib undoes it on
input (``remove_recording_time_offset``).
"""

import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from ._red import MEF3_RED_LAYOUT, decode_red_blocks
from .._channel_ids import _stacklevel_outside_neuroconv

UNIVERSAL_HEADER_BYTES = 1024
METADATA_FILE_BYTES = 16384
TIME_SERIES_INDEX_BYTES = 56
RED_BLOCK_FIXED_HEADER_BYTES = 48  # the header fields before the symbol statistics
RECORD_HEADER_BYTES = 24
RECORD_INDEX_BYTES = 24

# UUTC_NO_ENTRY, which is also METADATA_RECORDING_TIME_OFFSET_NO_ENTRY.
UUTC_NO_ENTRY = -(2**63)
GMT_OFFSET_NO_ENTRY = -86401
MEF_LITTLE_ENDIAN = 1

# RED_LEVEL_1_ENCRYPTION_MASK | RED_LEVEL_2_ENCRYPTION_MASK, in a block's flags.
_RED_ENCRYPTION_MASK = 0x06
# RED_DISCONTINUITY_MASK: the block does not continue the one before it.
_RED_DISCONTINUITY_MASK = 0x01
# An unflagged block whose start time is this far from where the samples before it end is taken to follow a gap.
_UNFLAGGED_GAP_MICROSECONDS = 1_000_000

# The fields of a .tidx entry (TIME_SERIES_INDEX_*_OFFSET).
_TIME_SERIES_INDEX_DTYPE = np.dtype(
    {
        "names": ["file_offset", "start_time", "start_sample", "number_of_samples", "block_bytes", "flags"],
        "formats": ["<i8", "<i8", "<i8", "<u4", "<u4", "u1"],
        "offsets": [0, 8, 16, 24, 28, 44],
        "itemsize": TIME_SERIES_INDEX_BYTES,
    }
)
# The fixed fields of a RED block header (RED_BLOCK_*_OFFSET).
_RED_BLOCK_HEADER_DTYPE = np.dtype(
    {
        "names": [
            "flags",
            "detrend_slope",
            "detrend_intercept",
            "scale_factor",
            "difference_bytes",
            "number_of_samples",
            "block_bytes",
            "start_time",
        ],
        "formats": ["u1", "<f4", "<f4", "<f4", "<u4", "<u4", "<u4", "<i8"],
        "offsets": [4, 16, 20, 24, 28, 32, 36, 40],
        "itemsize": RED_BLOCK_FIXED_HEADER_BYTES,
    }
)
_RECORD_INDEX_DTYPE = np.dtype(
    {
        "names": ["type", "encryption", "file_offset", "time"],
        "formats": ["S4", "i1", "<i8", "<i8"],
        "offsets": [0, 7, 8, 16],
        "itemsize": RECORD_INDEX_BYTES,
    }
)


@dataclass
class MEF3Channel:
    """One time series channel of a session: its metadata and the index of every block in every segment."""

    name: str
    directory: Path
    sampling_frequency: float
    units_conversion_factor: float
    units_description: str
    acquisition_channel_number: int
    reference_description: str
    data_paths: list[Path]
    # One entry per block with samples, in time order across segments: the segment it is in (an index into
    # data_paths), where it is in that segment's data file, its true start time in µUTC, and the fields of
    # its RED header the decoder needs.
    blocks: np.ndarray = field(repr=False)
    # The channel's own runs of blocks. Every channel shares the session's runs in time and samples, but not
    # necessarily its block boundaries.
    runs: list["MEF3Run"] = field(default_factory=list, repr=False)

    @property
    def number_of_samples(self) -> int:
        return int(self.blocks["number_of_samples"].sum())

    def read_blocks(self, first_block: int, number_of_blocks: int) -> np.ndarray:
        """Decode ``number_of_blocks`` consecutive blocks into one int32 array, one contiguous read per data file."""
        blocks = self.blocks[first_block : first_block + number_of_blocks]
        decoded = []
        for segment in np.unique(blocks["segment"]):
            in_segment = blocks[blocks["segment"] == segment]
            first_byte = int(in_segment["file_offset"][0])
            last_byte = int(in_segment["file_offset"][-1] + in_segment["block_bytes"][-1])
            with open(self.data_paths[segment], "rb") as file:
                file.seek(first_byte)
                buffer = np.frombuffer(file.read(last_byte - first_byte), dtype=np.uint8)
            if buffer.size != last_byte - first_byte:
                raise ValueError(
                    f"The MEF 3 data file '{self.data_paths[segment]}' ends before the blocks its index lists."
                )
            decoded.append(
                decode_red_blocks(
                    buffer,
                    block_starts=in_segment["file_offset"] - first_byte,
                    block_bytes=in_segment["block_bytes"],
                    difference_counts=in_segment["difference_bytes"],
                    sample_counts=in_segment["number_of_samples"],
                    scale_factors=in_segment["scale_factor"],
                    detrend_slopes=in_segment["detrend_slope"],
                    detrend_intercepts=in_segment["detrend_intercept"],
                    layout=MEF3_RED_LAYOUT,
                )
            )
        return np.concatenate(decoded) if len(decoded) > 1 else decoded[0]


@dataclass(frozen=True)
class MEF3Run:
    """One uninterrupted run of a channel's blocks. Every channel's runs agree in start time and samples."""

    first_block: int
    number_of_blocks: int
    number_of_samples: int
    start_time: int  # µUTC


@dataclass
class MEF3Record:
    """One MEF record (``Note``, ``Seiz``, ``ESti``, ...), as the rows of the records table need it."""

    time: int  # µUTC
    record_type: str
    label: str
    details: str
    duration: int  # µs, 0 for a record without one
    channel: str  # empty for a session-level record


@dataclass
class MEF3Session:
    """A MEF 3.0 session, as far as reading its samples needs."""

    path: Path
    channels: list[MEF3Channel]
    runs: list[MEF3Run]
    sampling_frequency: float
    gmt_offset: int | None  # seconds east of UTC, when the metadata gives one
    records: list[MEF3Record]
    lossy_blocks: dict[str, int]  # channel name to its number of scaled or detrended blocks
    largest_scale_factor: float

    @property
    def start_time(self) -> int:
        """Start of the first run, in µUTC."""
        return self.runs[0].start_time

    @property
    def start_datetime(self) -> datetime:
        """:attr:`start_time` as a datetime, in the session's own time zone when its metadata gives one."""
        start = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(microseconds=self.start_time)
        if self.gmt_offset is not None:
            start = start.astimezone(timezone(timedelta(seconds=self.gmt_offset)))
        return start


def _text(raw: bytes) -> str:
    """A NUL-terminated UTF-8 field."""
    return raw.split(b"\0", 1)[0].decode("utf-8", errors="replace").strip()


def _number(raw: bytes, offset: int, dtype: str):
    return np.frombuffer(raw, dtype=dtype, count=1, offset=offset)[0].item()


def _read_exactly(path: Path, number_of_bytes: int | None = None) -> bytes:
    with open(path, "rb") as file:
        raw = file.read() if number_of_bytes is None else file.read(number_of_bytes)
    if number_of_bytes is not None and len(raw) < number_of_bytes:
        raise ValueError(
            f"'{path}' holds {len(raw)} bytes, fewer than the {number_of_bytes} a MEF 3 file of its kind has."
        )
    return raw


def _check_universal_header(raw: bytes, path: Path, file_type: str):
    """Check the universal header (UNIVERSAL_HEADER_*_OFFSET) says this is a little-endian MEF 3 file of the expected type."""
    if len(raw) < UNIVERSAL_HEADER_BYTES:
        raise ValueError(f"'{path}' is shorter than a MEF universal header, so it is not a MEF 3 file.")
    stated_type = raw[8:12].decode("ascii", errors="replace")
    if stated_type != file_type:
        raise ValueError(f"'{path}' declares the MEF file type {stated_type!r}, not {file_type!r}.")
    major, minor, byte_order = raw[13], raw[14], raw[15]
    if major != 3:
        raise ValueError(f"'{path}' is a MEF {major}.{minor} file; only MEF 3 is supported here.")
    if byte_order != MEF_LITTLE_ENDIAN:
        raise ValueError(f"'{path}' is a big-endian MEF file, which is not supported.")


def _true_time(stored_time: int, recording_time_offset: int) -> int:
    """
    ``remove_recording_time_offset``: a negative stored time has the recording time offset applied.

    Works on scalars and on int64 arrays alike.
    """
    if recording_time_offset == UUTC_NO_ENTRY:
        recording_time_offset = 0
    stored_time = np.asarray(stored_time, dtype=np.int64)
    true_time = np.where(stored_time < 0, -stored_time + recording_time_offset, stored_time)
    return true_time if true_time.ndim else int(true_time)


def _segment_directories(channel_directory: Path) -> list[Path]:
    """The ``.segd`` directories of a channel, in segment order (the six-digit number in their names)."""

    def segment_number(directory: Path) -> int:
        number = directory.stem.rsplit("-", 1)[-1]
        if not number.isdigit():
            raise ValueError(f"The MEF 3 segment directory '{directory}' is not named '<channel>-<number>.segd'.")
        return int(number)

    return sorted((path for path in channel_directory.glob("*.segd") if path.is_dir()), key=segment_number)


def _only_file(directory: Path, suffix: str) -> Path:
    matches = sorted(directory.glob(f"*{suffix}"))
    if len(matches) != 1:
        raise ValueError(f"The MEF 3 segment directory '{directory}' holds {len(matches)} {suffix} files, not one.")
    return matches[0]


def _read_segment_metadata(path: Path) -> dict:
    raw = _read_exactly(path, METADATA_FILE_BYTES)
    _check_universal_header(raw, path, "tmet")
    # METADATA_SECTION_2_ENCRYPTION_OFFSET and _3_: a positive level is encrypted on disk; 0 is never
    # encrypted, and a negative level was decrypted before it was written.
    section_2_encryption, section_3_encryption = np.frombuffer(raw, dtype="i1", count=2, offset=1024)
    if section_2_encryption > 0 or section_3_encryption > 0:
        raise ValueError(
            f"The MEF 3 metadata file '{path}' is encrypted (section 2 at level {section_2_encryption}, section 3 at "
            f"level {section_3_encryption}). Encrypted MEF 3 sessions are not supported; decrypt the session with "
            "the MEF tools and its passwords first."
        )
    gmt_offset = _number(raw, 13336, "<i4")
    return dict(
        channel_name=_text(raw[52:308]),
        reference_description=_text(raw[6664:8712]),
        acquisition_channel_number=_number(raw, 8712, "<i8"),
        sampling_frequency=_number(raw, 8720, "<f8"),
        units_conversion_factor=_number(raw, 8760, "<f8"),
        units_description=_text(raw[8768:8896]),
        recording_time_offset=_number(raw, 13312, "<i8"),
        gmt_offset=None if gmt_offset == GMT_OFFSET_NO_ENTRY or abs(gmt_offset) > 86400 else gmt_offset,
    )


def _read_block_index(index_path: Path, data_path: Path, recording_time_offset: int) -> np.ndarray:
    """
    The blocks of one segment, with the fields of their RED headers.

    The headers are read from the data file one at a time: 48 bytes per block, a few thousand reads for a
    long channel, which keeps every read a byte range rather than a whole-file load.
    """
    raw = _read_exactly(index_path)
    _check_universal_header(raw, index_path, "tidx")
    body = raw[UNIVERSAL_HEADER_BYTES:]
    if len(body) % TIME_SERIES_INDEX_BYTES:
        raise ValueError(f"The MEF 3 index '{index_path}' is truncated mid-entry.")
    index = np.frombuffer(body, dtype=_TIME_SERIES_INDEX_DTYPE)

    headers = np.empty(len(index), dtype=_RED_BLOCK_HEADER_DTYPE)
    with open(data_path, "rb") as file:
        _check_universal_header(file.read(UNIVERSAL_HEADER_BYTES), data_path, "tdat")
        for position, offset in enumerate(index["file_offset"]):
            file.seek(int(offset))
            header = file.read(RED_BLOCK_FIXED_HEADER_BYTES)
            if len(header) < RED_BLOCK_FIXED_HEADER_BYTES:
                raise ValueError(
                    f"The MEF 3 data file '{data_path}' ends before block {position} that its index lists."
                )
            headers[position] = np.frombuffer(header, dtype=_RED_BLOCK_HEADER_DTYPE)[0]

    mismatched = np.flatnonzero(
        (headers["number_of_samples"] != index["number_of_samples"]) | (headers["block_bytes"] != index["block_bytes"])
    )
    if mismatched.size:
        raise ValueError(
            f"Block {mismatched[0]} of '{data_path}' does not have the size or number of samples its index "
            f"'{index_path.name}' gives it ({mismatched.size} such blocks). The session is corrupt."
        )
    encrypted = np.flatnonzero((headers["flags"] | index["flags"]) & _RED_ENCRYPTION_MASK)
    if encrypted.size:
        raise ValueError(
            f"{encrypted.size} blocks of '{data_path}' are encrypted. Encrypted MEF 3 sessions are not supported; "
            "decrypt the session with the MEF tools and its passwords first."
        )

    blocks = np.empty(
        len(index),
        dtype=[
            ("segment", "i8"),
            ("file_offset", "i8"),
            ("start_time", "i8"),
            ("number_of_samples", "i8"),
            ("block_bytes", "i8"),
            ("difference_bytes", "i8"),
            ("scale_factor", "f8"),
            ("detrend_slope", "f8"),
            ("detrend_intercept", "f8"),
            ("discontinuity", "?"),
        ],
    )
    blocks["file_offset"] = index["file_offset"]
    blocks["start_time"] = _true_time(index["start_time"], recording_time_offset)
    blocks["number_of_samples"] = index["number_of_samples"]
    blocks["block_bytes"] = index["block_bytes"]
    blocks["difference_bytes"] = headers["difference_bytes"]
    blocks["scale_factor"] = headers["scale_factor"]
    blocks["detrend_slope"] = headers["detrend_slope"]
    blocks["detrend_intercept"] = headers["detrend_intercept"]
    blocks["discontinuity"] = ((headers["flags"] | index["flags"]) & _RED_DISCONTINUITY_MASK).astype(bool)
    return blocks


def _read_channel(channel_directory: Path) -> tuple[MEF3Channel, int, int | None]:
    """A channel, its recording time offset and its GMT offset."""
    segment_directories = _segment_directories(channel_directory)
    if not segment_directories:
        raise ValueError(f"The MEF 3 channel directory '{channel_directory}' holds no .segd segment directories.")

    channel_metadata = None
    data_paths, all_blocks = [], []
    for segment, segment_directory in enumerate(segment_directories):
        metadata = _read_segment_metadata(_only_file(segment_directory, ".tmet"))
        if channel_metadata is None:
            channel_metadata = metadata
        elif metadata["sampling_frequency"] != channel_metadata["sampling_frequency"]:
            raise ValueError(
                f"The segments of the MEF 3 channel '{channel_directory.name}' are sampled at different rates "
                f"({channel_metadata['sampling_frequency']} and {metadata['sampling_frequency']} Hz)."
            )
        elif metadata["units_conversion_factor"] != channel_metadata["units_conversion_factor"]:
            raise ValueError(
                f"The segments of the MEF 3 channel '{channel_directory.name}' have different units conversion "
                f"factors ({channel_metadata['units_conversion_factor']} and {metadata['units_conversion_factor']})."
            )
        data_path = _only_file(segment_directory, ".tdat")
        blocks = _read_block_index(_only_file(segment_directory, ".tidx"), data_path, metadata["recording_time_offset"])
        blocks["segment"] = segment
        data_paths.append(data_path)
        all_blocks.append(blocks)

    blocks = np.concatenate(all_blocks)
    blocks = blocks[blocks["number_of_samples"] > 0]
    sampling_frequency = channel_metadata["sampling_frequency"]
    if not (np.isfinite(sampling_frequency) and sampling_frequency > 0):
        raise ValueError(
            f"The MEF 3 channel '{channel_directory.name}' gives a sampling frequency of {sampling_frequency}."
        )
    channel = MEF3Channel(
        # The directory names the channel, so names are unique within a session.
        name=channel_directory.name[: -len(".timd")],
        directory=channel_directory,
        sampling_frequency=sampling_frequency,
        units_conversion_factor=channel_metadata["units_conversion_factor"],
        units_description=channel_metadata["units_description"],
        acquisition_channel_number=channel_metadata["acquisition_channel_number"],
        reference_description=channel_metadata["reference_description"],
        data_paths=data_paths,
        blocks=blocks,
    )
    return channel, channel_metadata["recording_time_offset"], channel_metadata["gmt_offset"]


def group_blocks_into_runs(blocks: np.ndarray, sampling_frequency: float) -> tuple[list[MEF3Run], np.ndarray]:
    """
    Group consecutive blocks into uninterrupted runs, and say where unflagged gaps were found.

    A run breaks at a block flagged as a discontinuity, which is how meflib, and pymef's table of contents
    and time-based reads built on it, define one. Timestamps alone cannot: a recording's clock drifts
    against its nominal rate, and writers cut blocks on a fixed grid of times regardless. ds004696's blocks
    sit exactly 5 s apart but hold 10239 to 10263 samples, so comparing start times with sample counts,
    even in whole samples, splits a continuous recording into overlapping pieces.

    Two exceptions, both decided in whole samples (the distance between start times, rounded to the
    nearest sample, against the previous block's number of samples):

    * A flagged block that starts exactly where the previous one ended continues the run. Writers flag the
      first block of every segment whether or not anything is missing.
    * An unflagged block whose start time is more than a second from where the previous block ended starts a
      new run, since no clock drifts that far in one block; the writer left a gap unflagged.

    Returns
    -------
    runs : list of MEF3Run
    unflagged_gaps : numpy.ndarray of int
        Indices of the blocks that start a run without a discontinuity flag.
    """
    if len(blocks) == 0:
        return [], np.array([], dtype=np.int64)
    start_times = blocks["start_time"]
    counts = blocks["number_of_samples"]
    elapsed = np.diff(start_times)
    expected = counts[:-1] * (1e6 / sampling_frequency)
    contiguous = np.rint(elapsed * (sampling_frequency / 1e6)).astype(np.int64) == counts[:-1]
    flagged = blocks["discontinuity"][1:]
    unflagged_gap = ~flagged & (np.abs(elapsed - expected) > _UNFLAGGED_GAP_MICROSECONDS)
    breaks = np.flatnonzero((flagged & ~contiguous) | unflagged_gap) + 1
    edges = np.concatenate([[0], breaks, [len(blocks)]])
    runs = [
        MEF3Run(
            first_block=int(first),
            number_of_blocks=int(last - first),
            number_of_samples=int(counts[first:last].sum()),
            start_time=int(start_times[first]),
        )
        for first, last in zip(edges[:-1], edges[1:])
    ]
    return runs, np.flatnonzero(unflagged_gap) + 1


def _check_shared_layout(channels: list[MEF3Channel], sampling_frequency: float, session_path: Path) -> list[MEF3Run]:
    """The runs of the session, which every channel must share: one electrodes table spans every series."""
    reference = channels[0]
    runs, unflagged_gaps = group_blocks_into_runs(reference.blocks, sampling_frequency)
    reference.runs = runs
    if unflagged_gaps.size:
        jumps = [
            (reference.blocks["start_time"][index] - reference.blocks["start_time"][index - 1]) / 1e6
            - reference.blocks["number_of_samples"][index - 1] / sampling_frequency
            for index in unflagged_gaps[:3]
        ]
        shown = ", ".join(f"block {index} ({jump:+.3f} s)" for index, jump in zip(unflagged_gaps, jumps))
        warnings.warn(
            f"The timestamps of the MEF 3 channel '{reference.name}' of '{session_path.name}' jump by more than a "
            f"second at {unflagged_gaps.size} blocks that are not flagged as discontinuities: {shown}. Each is "
            "taken as a gap and starts a new ElectricalSeries.",
            UserWarning,
            stacklevel=_stacklevel_outside_neuroconv(),
        )
    half_sample = 0.5e6 / sampling_frequency
    for channel in channels[1:]:
        channel_runs, _ = group_blocks_into_runs(channel.blocks, sampling_frequency)
        channel.runs = channel_runs
        same = len(channel_runs) == len(runs) and all(
            ours.number_of_samples == theirs.number_of_samples
            and abs(ours.start_time - theirs.start_time) < half_sample
            for ours, theirs in zip(runs, channel_runs)
        )
        if not same:
            raise ValueError(
                f"The MEF 3 channels '{reference.name}' and '{channel.name}' of '{session_path.name}' were not recorded "
                f"over the same stretches of time ({len(runs)} runs of {[run.number_of_samples for run in runs][:5]} "
                f"samples against {len(channel_runs)} runs of {[run.number_of_samples for run in channel_runs][:5]}). "
                "Convert them separately, leaving the others out with channels_to_skip."
            )
    return runs


def _record_body_fields(record_type: str, body: bytes) -> tuple[str, str, int]:
    """
    The label, details and duration (µs) of a record, from its body (``mefrec.h``, version 1.0 of each type).

    Types this does not know are kept with their type alone.
    """

    def text(start: int, length: int | None = None) -> str:
        return _text(body[start : None if length is None else start + length])

    def integer(start: int, dtype: str = "<i8"):
        size = np.dtype(dtype).itemsize
        return _number(body, start, dtype) if len(body) >= start + size else 0

    def join(**fields) -> str:
        return "; ".join(f"{name}={value}" for name, value in fields.items() if value not in ("", None))

    if record_type in ("Note", "SyLg"):
        return text(0), "", 0
    if record_type == "EDFA":
        return text(8), "", max(integer(0), 0)
    if record_type == "Seiz":
        details = join(
            earliest_onset=integer(0),
            latest_offset=integer(8),
            number_of_channels=integer(24, "<i4"),
            onset_code=integer(28, "<i4"),
            marker_name_1=text(32, 128),
            marker_name_2=text(160, 128),
        )
        return text(288, 1024), details, max(integer(16), 0)
    if record_type == "CSti":
        details = join(task_type=text(0, 64), patient_response=text(136, 64))
        return text(72, 64), details, max(integer(64), 0)
    if record_type == "ESti":
        amplitude_units = {1: "mA", 2: "V"}.get(integer(24, "<i4"), "")
        mode = {1: "current", 2: "voltage"}.get(integer(28, "<i4"), "")
        details = join(
            amplitude=f"{integer(0, '<f8')} {amplitude_units}".strip(),
            frequency_hz=integer(8, "<f8"),
            pulse_width_us=integer(16),
            mode=mode,
            anode=text(160, 128),
            cathode=text(288, 128),
        )
        return text(32, 128), details, 0
    if record_type == "Curs":
        details = join(id_number=integer(0), trace_timestamp=integer(8), latency=integer(16), value=integer(24, "<f8"))
        return text(32, 128), details, 0
    if record_type == "Epoc":
        details = join(id_number=integer(0), epoch_type=text(32, 32))
        return text(64, 128), details, max(integer(24), 0)
    if record_type == "LNTP":
        return "", join(template_length=integer(0)), 0
    return "", "", 0


def _read_records(directory: Path, channel: str, recording_time_offset: int) -> tuple[list[MEF3Record], int]:
    """The records of one level of the session, and how many were left out for being encrypted."""
    records, encrypted = [], 0
    for index_path in sorted(directory.glob("*.ridx")):
        data_path = index_path.with_suffix(".rdat")
        if not data_path.is_file():
            warnings.warn(
                f"The MEF 3 record index '{index_path}' has no .rdat beside it, so its records are left out.",
                UserWarning,
                stacklevel=_stacklevel_outside_neuroconv(),
            )
            continue
        raw_index = _read_exactly(index_path)
        _check_universal_header(raw_index, index_path, "ridx")
        body = raw_index[UNIVERSAL_HEADER_BYTES:]
        entries = np.frombuffer(body[: len(body) - len(body) % RECORD_INDEX_BYTES], dtype=_RECORD_INDEX_DTYPE)
        raw_data = _read_exactly(data_path)
        _check_universal_header(raw_data, data_path, "rdat")
        for entry in entries:
            offset = int(entry["file_offset"])
            header = raw_data[offset : offset + RECORD_HEADER_BYTES]
            if len(header) < RECORD_HEADER_BYTES:
                continue
            # RECORD_HEADER_ENCRYPTION_OFFSET; positive is encrypted on disk.
            if np.frombuffer(header, dtype="i1", count=1, offset=11)[0] > 0:
                encrypted += 1
                continue
            record_type = header[4:8].decode("ascii", errors="replace")
            body_bytes = _number(header, 12, "<u4")
            record_body = raw_data[offset + RECORD_HEADER_BYTES : offset + RECORD_HEADER_BYTES + body_bytes]
            label, details, duration = _record_body_fields(record_type, record_body)
            records.append(
                MEF3Record(
                    time=_true_time(_number(header, 16, "<i8"), recording_time_offset),
                    record_type=record_type,
                    label=label,
                    details=details,
                    duration=duration,
                    channel=channel,
                )
            )
    return records, encrypted


def read_mef3_session(folder_path, channels_to_skip: list | None = None) -> MEF3Session:
    """
    Read the metadata and block indices of a MEF 3.0 session, without decoding any samples.

    Parameters
    ----------
    folder_path : str or Path
        The session's ``.mefd`` directory.
    channels_to_skip : list of str, optional
        Names of channels to leave out. Their files are not read, so a channel sampled at another rate, or
        over other stretches of time, can be dropped before the checks that every kept channel agrees.

    Returns
    -------
    MEF3Session
        Channels are in acquisition channel order when every channel gives a distinct one, else by name.
    """
    path = Path(folder_path)
    if not path.is_dir():
        raise ValueError(f"'{path}' is not a directory. A MEF 3 session is a .mefd directory.")
    channel_directories = sorted(directory for directory in path.glob("*.timd") if directory.is_dir())
    if not channel_directories:
        raise ValueError(f"'{path}' holds no .timd channel directories, so it is not a MEF 3 session.")
    names = [directory.name[: -len(".timd")] for directory in channel_directories]
    unknown = sorted(set(channels_to_skip or []) - set(names))
    if unknown:
        raise ValueError(f"channels_to_skip names channels that are not in the session: {unknown}. Available: {names}.")
    channel_directories = [
        directory for directory, name in zip(channel_directories, names) if name not in set(channels_to_skip or [])
    ]
    if not channel_directories:
        raise ValueError("channels_to_skip leaves no channels in the session.")

    channels, recording_time_offsets, gmt_offsets = [], [], []
    for directory in channel_directories:
        channel, recording_time_offset, gmt_offset = _read_channel(directory)
        if channel.number_of_samples == 0:
            raise ValueError(
                f"The MEF 3 channel '{channel.name}' of '{path.name}' holds no samples. Leave it out with "
                "channels_to_skip."
            )
        channels.append(channel)
        recording_time_offsets.append(recording_time_offset)
        gmt_offsets.append(gmt_offset)

    order = _channel_order(
        [channel.name for channel in channels], [channel.acquisition_channel_number for channel in channels]
    )
    channels = [channels[index] for index in order]

    rates = {}
    for channel in channels:
        rates.setdefault(channel.sampling_frequency, []).append(channel.name)
    if len(rates) > 1:
        described = "; ".join(f"{rate} Hz: {', '.join(names[:5])}" for rate, names in sorted(rates.items()))
        raise ValueError(
            f"The channels of the MEF 3 session '{path.name}' are sampled at {len(rates)} different rates ({described}). "
            "An ElectricalSeries has one rate, so convert each rate separately, leaving the others out with "
            "channels_to_skip."
        )
    sampling_frequency = channels[0].sampling_frequency
    runs = _check_shared_layout(channels, sampling_frequency, path)

    # The session-level records use the recording time offset the channels share; MEF 3 writers give every
    # segment the session's.
    session_recording_time_offset = recording_time_offsets[0]
    records, encrypted = _read_records(path, channel="", recording_time_offset=session_recording_time_offset)
    for channel, recording_time_offset in zip(channels, recording_time_offsets):
        channel_records, channel_encrypted = _read_records(channel.directory, channel.name, recording_time_offset)
        records += channel_records
        encrypted += channel_encrypted
        for segment_directory in _segment_directories(channel.directory):
            segment_records, segment_encrypted = _read_records(segment_directory, channel.name, recording_time_offset)
            records += segment_records
            encrypted += segment_encrypted
    if encrypted:
        warnings.warn(
            f"{encrypted} records of the MEF 3 session '{path.name}' are encrypted and are left out.",
            UserWarning,
            stacklevel=_stacklevel_outside_neuroconv(),
        )
    records.sort(key=lambda record: record.time)

    lossy_blocks, largest_scale_factor = {}, 1.0
    for channel in channels:
        blocks = channel.blocks
        lossy = (blocks["scale_factor"] > 1.0) | (blocks["detrend_slope"] != 0.0) | (blocks["detrend_intercept"] != 0.0)
        if lossy.any():
            lossy_blocks[channel.name] = int(lossy.sum())
            largest_scale_factor = max(largest_scale_factor, float(blocks["scale_factor"].max()))

    return MEF3Session(
        path=path,
        channels=channels,
        runs=runs,
        sampling_frequency=sampling_frequency,
        gmt_offset=gmt_offsets[0],
        records=records,
        lossy_blocks=lossy_blocks,
        largest_scale_factor=largest_scale_factor,
    )


def _channel_order(names: list[str], acquisition_channel_numbers: list[int]) -> list[int]:
    """Positions in acquisition channel order when every channel gives a distinct number, else in name order."""
    numbers = list(acquisition_channel_numbers)
    if all(number >= 0 for number in numbers) and len(set(numbers)) == len(numbers):
        return [int(index) for index in np.argsort(numbers, kind="stable")]
    return sorted(range(len(names)), key=lambda index: names[index])


def available_channel_names(folder_path) -> list[str]:
    """
    The names of a session's time series channels, in the order :func:`read_mef3_session` gives them.

    Only the first segment's metadata file of each channel is read.
    """
    path = Path(folder_path)
    channel_directories = sorted(directory for directory in path.glob("*.timd") if directory.is_dir())
    names, numbers = [], []
    for directory in channel_directories:
        segment_directories = _segment_directories(directory)
        if not segment_directories:
            raise ValueError(f"The MEF 3 channel directory '{directory}' holds no .segd segment directories.")
        names.append(directory.name[: -len(".timd")])
        numbers.append(
            _read_segment_metadata(_only_file(segment_directories[0], ".tmet"))["acquisition_channel_number"]
        )
    return [names[index] for index in _channel_order(names, numbers)]
