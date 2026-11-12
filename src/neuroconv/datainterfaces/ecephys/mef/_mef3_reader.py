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

A session whose metadata is encrypted is read with its password (see :mod:`._mef3_crypto`). Section 2 is
decrypted whole. Of section 3 only its first two AES blocks are decrypted, which hold the recording time offset,
the daylight saving times and the GMT offset, and the four bytes of ``subject_name_1`` that share the second
block are dropped at once; the rest of section 3 stays encrypted. Encrypted RED data blocks are not supported.

A session's channels may be sampled at different rates, or recorded over different stretches of time (a
continuous telemetry channel beside channels with gaps). They are grouped by rate and runs, each group being
what one ElectricalSeries can hold; see :class:`MEF3ChannelGroup`. A segment that holds no data file and no
index entry is left out.

Times in MEF are microseconds since the Unix epoch (µUTC). A writer may store them offset by the session's
recording time offset, marking such a time by making it negative; it is undone here as meflib undoes it on
input (``remove_recording_time_offset``). A record file whose writer offset its times by the session's start
instead is read from that start (see ``_records_clock``).
"""

import warnings
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from ._mef3_crypto import AES_BLOCK_BYTES, MEF3Passwords, aes_ecb_decrypt
from ._red import MEF3_RED_LAYOUT, decode_red_blocks
from .._channel_ids import _stacklevel_outside_neuroconv

UNIVERSAL_HEADER_BYTES = 1024
METADATA_FILE_BYTES = 16384
TIME_SERIES_INDEX_BYTES = 56
RED_BLOCK_FIXED_HEADER_BYTES = 48  # the header fields before the symbol statistics
RECORD_HEADER_BYTES = 24
RECORD_INDEX_BYTES = 24

# METADATA_SECTION_2_OFFSET and METADATA_SECTION_3_OFFSET; section 2 ends where section 3 starts.
_SECTION_2_START = 2560
_SECTION_3_START = 13312
# The fields read from section 3 are in its first 28 bytes: METADATA_RECORDING_TIME_OFFSET_OFFSET (13312, si8),
# DST start and end times (13320, 13328; not read) and METADATA_GMT_OFFSET_OFFSET (13336, si4). They span two
# AES blocks; subject_name_1 starts at 13340.
_SECTION_3_TIMES_BYTES = 28
_SECTION_3_TIMES_BLOCKS = 2

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
    # The times the record's body states besides its own (a seizure's earliest onset, a cursor's trace time), by
    # name, as true µUTC. They are kept out of ``details`` so that a writer can place them on its own timeline.
    times: dict[str, int] = field(default_factory=dict)


@dataclass
class MEF3ChannelGroup:
    """
    Channels of a session that one ElectricalSeries can hold: sampled at one rate and recorded over the same
    runs, in start time and samples.
    """

    channels: list[MEF3Channel]
    runs: list[MEF3Run]
    sampling_frequency: float

    def describe(self) -> str:
        names = ", ".join(channel.name for channel in self.channels[:5])
        if len(self.channels) > 5:
            names += f", ... ({len(self.channels)} channels)"
        samples = [run.number_of_samples for run in self.runs[:5]]
        return f"{self.sampling_frequency:g} Hz, {len(self.runs)} runs of {samples} samples: {names}"


@dataclass
class MEF3Session:
    """
    A MEF 3.0 session, as far as reading its samples needs.

    Its channels fall into one or more :class:`MEF3ChannelGroup`, each of which one ElectricalSeries can
    hold. :attr:`runs` and :attr:`sampling_frequency` are those of the only group, and raise when there are
    several; :meth:`select_group` gives a session of one group.
    """

    path: Path
    channels: list[MEF3Channel]
    # The largest group first (most channels, then the highest rate), then in the order of their first channels.
    channel_groups: list[MEF3ChannelGroup]
    gmt_offset: int | None  # seconds east of UTC, when the metadata gives one
    records: list[MEF3Record]
    lossy_blocks: dict[str, int]  # channel name to its number of scaled or detrended blocks
    largest_scale_factor: float
    # The session's start, in µUTC: the earliest first block of any of its channels, kept or skipped, so every
    # reader of one session over complementary channels_to_skip puts its series on one timeline.
    start_time: int
    # Every sampling rate in the session, skipped channels' included.
    session_sampling_frequencies: tuple[float, ...]

    def _only_group(self) -> MEF3ChannelGroup:
        if len(self.channel_groups) == 1:
            return self.channel_groups[0]
        described = "; ".join(f"({group.describe()})" for group in self.channel_groups)
        raise ValueError(
            f"The channels of the MEF 3 session '{self.path.name}' fall into {len(self.channel_groups)} groups "
            f"that are sampled at different rates or were not recorded over the same stretches of time: "
            f"{described}. An ElectricalSeries holds one group: MEF3RecordingInterface writes each as its own "
            "series, or keep one with channels_to_skip."
        )

    @property
    def runs(self) -> list[MEF3Run]:
        """The runs of the session's only channel group."""
        return self._only_group().runs

    @property
    def sampling_frequency(self) -> float:
        """The sampling frequency of the session's only channel group."""
        return self._only_group().sampling_frequency

    def select_group(self, index: int) -> "MEF3Session":
        """
        The session as one of its channel groups: its channels, runs and lossy blocks, with the session's start
        time, time zone and records.
        """
        group = self.channel_groups[index]
        names = {channel.name for channel in group.channels}
        lossy_blocks = {name: count for name, count in self.lossy_blocks.items() if name in names}
        largest_scale_factor = max(
            [1.0]
            + [
                float(channel.blocks["scale_factor"].max())
                for channel in group.channels
                if channel.name in lossy_blocks
            ]
        )
        return replace(
            self,
            channels=list(group.channels),
            channel_groups=[group],
            lossy_blocks=lossy_blocks,
            largest_scale_factor=largest_scale_factor,
        )

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
    ``UUTC_NO_ENTRY`` states no time and is returned as it is, as meflib returns it.

    Works on scalars and on int64 arrays alike.
    """
    if recording_time_offset == UUTC_NO_ENTRY:
        recording_time_offset = 0
    stored_time = np.asarray(stored_time, dtype=np.int64)
    offset = (stored_time < 0) & (stored_time != UUTC_NO_ENTRY)
    true_time = np.where(offset, -np.where(offset, stored_time, 0) + recording_time_offset, stored_time)
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


def _is_empty_segment(segment_directory: Path) -> bool:
    """
    Whether a segment holds no data file and no index entry: one a writer opened and never wrote a block to.

    A segment with no data file whose index lists blocks is not empty but broken, and is left for the reader
    to refuse.
    """
    if any(segment_directory.glob("*.tdat")):
        return False
    indices = sorted(segment_directory.glob("*.tidx"))
    if len(indices) > 1:
        return False
    return not indices or indices[0].stat().st_size < UNIVERSAL_HEADER_BYTES + TIME_SERIES_INDEX_BYTES


_LEAVE_EMPTY_CHANNELS_OUT = (
    "Leave it out with channels_to_skip; MEF3RecordingInterface.get_empty_channel_ids lists every channel of "
    "the session that holds no samples."
)


def _data_segment_directories(channel_directory: Path) -> tuple[list[Path], list[Path]]:
    """A channel's segment directories in segment order, as those that hold data and those that are empty."""
    data, empty = [], []
    for segment_directory in _segment_directories(channel_directory):
        (empty if _is_empty_segment(segment_directory) else data).append(segment_directory)
    return data, empty


def _read_segment_metadata(path: Path, passwords: MEF3Passwords | None = None) -> dict:
    raw = _read_exactly(path, METADATA_FILE_BYTES)
    _check_universal_header(raw, path, "tmet")
    # METADATA_SECTION_2_ENCRYPTION_OFFSET and _3_: a positive level is encrypted on disk; 0 is never
    # encrypted, and a negative level was decrypted before it was written.
    section_2_encryption, section_3_encryption = (
        int(level) for level in np.frombuffer(raw, dtype="i1", count=2, offset=1024)
    )
    section_2 = raw[_SECTION_2_START:_SECTION_3_START]
    section_3_times = raw[_SECTION_3_START : _SECTION_3_START + _SECTION_3_TIMES_BYTES]
    if section_2_encryption > 0 or section_3_encryption > 0:
        if passwords is None:
            raise ValueError(
                f"The MEF 3 metadata file '{path}' is encrypted (section 2 at level {section_2_encryption}, section 3 "
                f"at level {section_3_encryption}). Give the session's password to read it."
            )
        levels = {level for level in (section_2_encryption, section_3_encryption) if level > 0}
        if not levels <= {1, 2}:
            raise ValueError(f"The MEF 3 metadata file '{path}' gives an encryption level other than 1 or 2.")
        keys = passwords.keys(raw[:UNIVERSAL_HEADER_BYTES], path, levels)
        if section_2_encryption > 0:
            section_2 = aes_ecb_decrypt(keys[section_2_encryption], section_2)
        if section_3_encryption > 0:
            # Only the blocks that hold the times, and of those only the times are kept.
            encrypted_times = raw[_SECTION_3_START : _SECTION_3_START + _SECTION_3_TIMES_BLOCKS * AES_BLOCK_BYTES]
            section_3_times = aes_ecb_decrypt(keys[section_3_encryption], encrypted_times)[:_SECTION_3_TIMES_BYTES]

    def section_2_text(start: int, stop: int) -> str:
        return _text(section_2[start - _SECTION_2_START : stop - _SECTION_2_START])

    def section_2_number(offset: int, dtype: str):
        return _number(section_2, offset - _SECTION_2_START, dtype)

    # The daylight saving fields (13320, 13328) are not read. Some writers fill them with 0x0000800000000000,
    # a misplaced "no entry" that reads as 1974-06-17.
    gmt_offset = _number(section_3_times, 13336 - _SECTION_3_START, "<i4")
    return dict(
        channel_name=_text(raw[52:308]),
        reference_description=section_2_text(6664, 8712),
        acquisition_channel_number=section_2_number(8712, "<i8"),
        sampling_frequency=section_2_number(8720, "<f8"),
        units_conversion_factor=section_2_number(8760, "<f8"),
        units_description=section_2_text(8768, 8896),
        recording_time_offset=_number(section_3_times, 0, "<i8"),
        gmt_offset=None if gmt_offset == GMT_OFFSET_NO_ENTRY or abs(gmt_offset) > 86400 else gmt_offset,
    )


def _read_time_series_index(index_path: Path) -> np.ndarray:
    """The entries of a segment's .tidx, as stored (start times not yet made true)."""
    raw = _read_exactly(index_path)
    _check_universal_header(raw, index_path, "tidx")
    body = raw[UNIVERSAL_HEADER_BYTES:]
    if len(body) % TIME_SERIES_INDEX_BYTES:
        raise ValueError(f"The MEF 3 index '{index_path}' is truncated mid-entry.")
    return np.frombuffer(body, dtype=_TIME_SERIES_INDEX_DTYPE)


def _read_block_index(index_path: Path, data_path: Path, recording_time_offset: int) -> np.ndarray:
    """
    The blocks of one segment, with the fields of their RED headers.

    The headers are read from the data file one at a time: 48 bytes per block, a few thousand reads for a
    long channel, which keeps every read a byte range rather than a whole-file load.
    """
    index = _read_time_series_index(index_path)

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
            f"{encrypted.size} blocks of '{data_path}' are encrypted. Encrypted MEF 3 data blocks are not supported, "
            "with or without a password (encrypted metadata is); decrypt the session with the MEF tools first."
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


def _read_channel(
    channel_directory: Path, passwords: MEF3Passwords | None = None
) -> tuple[MEF3Channel, int, int | None]:
    """
    A channel, its recording time offset and its GMT offset.

    Empty segments, with no data file and no index entry, are left out with a warning.
    """
    if not _segment_directories(channel_directory):
        raise ValueError(f"The MEF 3 channel directory '{channel_directory}' holds no .segd segment directories.")
    segment_directories, empty_segments = _data_segment_directories(channel_directory)
    if not segment_directories:
        raise ValueError(
            f"Every segment of the MEF 3 channel '{channel_directory.name}' is empty, with no data file and no "
            f"block in its index, so the channel holds no samples. {_LEAVE_EMPTY_CHANNELS_OUT}"
        )
    if empty_segments:
        warnings.warn(
            f"{len(empty_segments)} segments of the MEF 3 channel '{channel_directory.name}' hold no data file and "
            f"no block in their index, and are left out: {', '.join(path.name for path in empty_segments[:5])}.",
            UserWarning,
            stacklevel=_stacklevel_outside_neuroconv(),
        )

    channel_metadata = None
    data_paths, all_blocks = [], []
    for segment, segment_directory in enumerate(segment_directories):
        metadata = _read_segment_metadata(_only_file(segment_directory, ".tmet"), passwords)
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


def group_blocks_into_runs(
    blocks: np.ndarray, sampling_frequency: float, *, backward_jumps_break: bool = True
) -> tuple[list[MEF3Run], np.ndarray]:
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

    The MEF 2 reader shares this rule with ``backward_jumps_break=False``: there an unflagged block that starts
    more than a second *before* the previous one ended continues the run. MEF 2 writers re-synchronise block
    times to the wall clock in steps, which can go back by a second while the samples run on without a break.

    Returns
    -------
    runs : list of MEF3Run
    unflagged_jumps : numpy.ndarray of int
        Indices of the unflagged blocks that start more than a second from where the previous block ended.
        Each starts a run, except a backward one when ``backward_jumps_break`` is False.
    """
    if len(blocks) == 0:
        return [], np.array([], dtype=np.int64)
    start_times = blocks["start_time"]
    counts = blocks["number_of_samples"]
    elapsed = np.diff(start_times)
    expected = counts[:-1] * (1e6 / sampling_frequency)
    contiguous = np.rint(elapsed * (sampling_frequency / 1e6)).astype(np.int64) == counts[:-1]
    flagged = blocks["discontinuity"][1:]
    unflagged_jump = ~flagged & (np.abs(elapsed - expected) > _UNFLAGGED_GAP_MICROSECONDS)
    unflagged_break = unflagged_jump if backward_jumps_break else unflagged_jump & (elapsed > expected)
    breaks = np.flatnonzero((flagged & ~contiguous) | unflagged_break) + 1
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
    return runs, np.flatnonzero(unflagged_jump) + 1


def _warn_about_unflagged_gaps(channel: MEF3Channel, unflagged_gaps: np.ndarray, session_path: Path):
    if not unflagged_gaps.size:
        return
    blocks, sampling_frequency = channel.blocks, channel.sampling_frequency
    jumps = [
        (blocks["start_time"][index] - blocks["start_time"][index - 1]) / 1e6
        - blocks["number_of_samples"][index - 1] / sampling_frequency
        for index in unflagged_gaps[:3]
    ]
    shown = ", ".join(f"block {index} ({jump:+.3f} s)" for index, jump in zip(unflagged_gaps, jumps))
    warnings.warn(
        f"The timestamps of the MEF 3 channel '{channel.name}' of '{session_path.name}' jump by more than a "
        f"second at {unflagged_gaps.size} blocks that are not flagged as discontinuities: {shown}. Each is "
        "taken as a gap and starts a new ElectricalSeries.",
        UserWarning,
        stacklevel=_stacklevel_outside_neuroconv(),
    )


def _same_runs(ours: list[MEF3Run], theirs: list[MEF3Run], sampling_frequency: float) -> bool:
    """Whether two channels at one rate were recorded over the same runs: the same samples, starting within half a sample."""
    half_sample = 0.5e6 / sampling_frequency
    return len(ours) == len(theirs) and all(
        mine.number_of_samples == other.number_of_samples and abs(mine.start_time - other.start_time) < half_sample
        for mine, other in zip(ours, theirs)
    )


def _group_channels(channels: list[MEF3Channel], session_path: Path) -> list[MEF3ChannelGroup]:
    """
    Group channels, in their order, by sampling rate and runs: each group is what one ElectricalSeries can hold.

    Each channel's runs are found with :func:`group_blocks_into_runs`; a channel joins the first group at its
    rate whose runs it shares, else starts its own. Unflagged gaps are warned about for each group's first
    channel. Channels at one rate that fall into several groups are warned about too: they were recorded over
    different stretches of time, which is how a session holds both a continuous channel and one with gaps.
    """
    groups: list[MEF3ChannelGroup] = []
    for channel in channels:
        runs, unflagged_gaps = group_blocks_into_runs(channel.blocks, channel.sampling_frequency)
        channel.runs = runs
        for group in groups:
            if group.sampling_frequency == channel.sampling_frequency and _same_runs(
                group.runs, runs, channel.sampling_frequency
            ):
                group.channels.append(channel)
                break
        else:
            _warn_about_unflagged_gaps(channel, unflagged_gaps, session_path)
            groups.append(
                MEF3ChannelGroup(channels=[channel], runs=runs, sampling_frequency=channel.sampling_frequency)
            )

    by_rate = {}
    for group in groups:
        by_rate.setdefault(group.sampling_frequency, []).append(group)
    for rate, groups_at_rate in by_rate.items():
        if len(groups_at_rate) > 1:
            described = "; ".join(f"({group.describe()})" for group in groups_at_rate)
            warnings.warn(
                f"The MEF 3 channels of '{session_path.name}' sampled at {rate:g} Hz were not all recorded over the "
                f"same stretches of time, so they are read as {len(groups_at_rate)} groups, each written as its own "
                f"series: {described}.",
                UserWarning,
                stacklevel=_stacklevel_outside_neuroconv(),
            )

    position = {channel.name: index for index, channel in enumerate(channels)}
    return sorted(
        groups,
        key=lambda group: (-len(group.channels), -group.sampling_frequency, position[group.channels[0].name]),
    )


def _record_body_fields(record_type: str, body: bytes) -> tuple[str, str, int, dict[str, int]]:
    """
    The label, details, duration (µs) and body times of a record, from its body (``mefrec.h``, version 1.0 of
    each type).

    The body times are the µUTC times the body states (``Seiz`` ``earliest_onset`` and ``latest_offset``,
    ``Curs`` ``trace_timestamp``), as stored: a time the recording time offset was applied to is negative, as
    in every MEF 3 time field. They are returned apart from ``details``, which holds the other fields, so the
    caller can recover their true value and place them on a timeline. A time stored as ``UUTC_NO_ENTRY`` or as
    0 (a body its writer left zero-filled) is left out: it states no time.

    Types this does not know are kept with their type alone.
    """

    def text(start: int, length: int | None = None) -> str:
        return _text(body[start : None if length is None else start + length])

    def integer(start: int, dtype: str = "<i8"):
        size = np.dtype(dtype).itemsize
        return _number(body, start, dtype) if len(body) >= start + size else 0

    def join(**fields) -> str:
        return "; ".join(f"{name}={value}" for name, value in fields.items() if value not in ("", None))

    def times(**fields) -> dict[str, int]:
        return {name: value for name, value in fields.items() if value not in (UUTC_NO_ENTRY, 0)}

    if record_type in ("Note", "SyLg"):
        return text(0), "", 0, {}
    if record_type == "EDFA":
        return text(8), "", max(integer(0), 0), {}
    if record_type == "Seiz":
        body_times = times(earliest_onset=integer(0), latest_offset=integer(8))
        details = join(
            number_of_channels=integer(24, "<i4"),
            onset_code=integer(28, "<i4"),
            marker_name_1=text(32, 128),
            marker_name_2=text(160, 128),
        )
        return text(288, 1024), details, max(integer(16), 0), body_times
    if record_type == "CSti":
        details = join(task_type=text(0, 64), patient_response=text(136, 64))
        return text(72, 64), details, max(integer(64), 0), {}
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
        return text(32, 128), details, 0, {}
    if record_type == "Curs":
        details = join(id_number=integer(0), latency=integer(16), value=integer(24, "<f8"))
        return text(32, 128), details, 0, times(trace_timestamp=integer(8))
    if record_type == "Epoc":
        details = join(id_number=integer(0), epoch_type=text(32, 32))
        return text(64, 128), details, max(integer(24), 0), {}
    if record_type == "LNTP":
        return "", join(template_length=integer(0)), 0, {}
    return "", "", 0, {}


# How far past the session's last sample a record read from the session's start may fall (see _records_clock).
_SESSION_CLOCK_TOLERANCE = 1_000_000  # µs


def _records_clock(stored_times: np.ndarray, recording_time_offset: int, session_span: tuple[int, int] | None):
    """
    The time a record file's offset times count from: the recording time offset, as MEF 3 has it, or the
    session's start, for a writer that stored its records offset by the session's start instead.

    Read with a recording time offset far from the start, such records all fall far outside the session,
    decades before it; read from the session's start, every one falls within it. The start is taken only when
    the two readings cannot be confused: every time in the file is stored offset (negative, or 0 at the start
    itself), every one read with the recording time offset falls outside the session by more than the
    session's own length, and every one read from the start falls within the session. An offset near the
    start, as most writers set it, therefore never moves a file. ``UUTC_NO_ENTRY`` states no time and is left
    out of the decision.

    The format leaves two cases ambiguous, and both keep the file as stored: a record that such a writer placed
    before the session's start is stored positive, as meflib stores any time before the offset; and a record
    more than ``_SESSION_CLOCK_TOLERANCE`` past the session's last sample (a late "recording stopped" note)
    does not fit the session from its start.
    """
    stored_times = np.asarray(stored_times, dtype=np.int64)
    stored_times = stored_times[stored_times != UUTC_NO_ENTRY]
    if session_span is None or len(stored_times) == 0 or np.any(stored_times > 0):
        return recording_time_offset
    start, end = session_span
    length = end - start
    # A time stored as 0 is the recording time offset itself (_records_true_times), so an offset near the start
    # keeps a file of zeros as stored.
    as_stated = _records_true_times(stored_times, recording_time_offset)
    from_start = -stored_times + start
    apart = (as_stated < start - length) | (as_stated > end + length + _SESSION_CLOCK_TOLERANCE)
    if np.all(apart) and np.all(from_start <= end + _SESSION_CLOCK_TOLERANCE):
        return start
    return recording_time_offset


def _records_true_times(stored_times, clock: int):
    """
    The true µUTC times of a record file's stored times, read with ``clock``: the recording time offset, or the
    session's start (see :func:`_records_clock`). A time stored as 0 is the clock itself: meflib's
    ``apply_recording_time_offset`` stores a time ``t`` as ``-(t - offset)``, so a record written at the offset
    is stored as 0, which its ``remove_recording_time_offset`` would read as the epoch. Works on scalars and on
    int64 arrays alike.
    """
    true_times = _true_time(stored_times, clock)
    stored = np.asarray(stored_times, dtype=np.int64)
    true_times = np.where(stored == 0, 0 if clock == UUTC_NO_ENTRY else clock, true_times)
    return true_times if true_times.ndim else int(true_times)


def _read_records(
    directory: Path, channel: str, recording_time_offset: int, session_span: tuple[int, int] | None = None
) -> tuple[list[MEF3Record], int, int]:
    """
    The records of one level of the session, how many were left out for being encrypted, and how many were read
    from the session's start rather than with the recording time offset (see :func:`_records_clock`).

    ``session_span`` is the session's first and last µUTC time, from the samples of every channel, kept or
    skipped, so the records do not depend on which channels are kept.
    """
    records, encrypted, records_from_start = [], 0, 0
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
        readable = []
        for entry in entries:
            offset = int(entry["file_offset"])
            header = raw_data[offset : offset + RECORD_HEADER_BYTES]
            if len(header) < RECORD_HEADER_BYTES:
                continue
            # RECORD_HEADER_ENCRYPTION_OFFSET; positive is encrypted on disk.
            if np.frombuffer(header, dtype="i1", count=1, offset=11)[0] > 0:
                encrypted += 1
                continue
            readable.append((offset, header))
        stored_times = np.array([_number(header, 16, "<i8") for _, header in readable], dtype=np.int64)
        clock = _records_clock(stored_times, recording_time_offset, session_span)
        if clock != recording_time_offset:
            records_from_start += len(readable)
        true_times = _records_true_times(stored_times, clock)

        for (offset, header), true_time in zip(readable, true_times):
            record_type = header[4:8].decode("ascii", errors="replace")
            body_bytes = _number(header, 12, "<u4")
            record_body = raw_data[offset + RECORD_HEADER_BYTES : offset + RECORD_HEADER_BYTES + body_bytes]
            label, details, duration, body_times = _record_body_fields(record_type, record_body)
            records.append(
                MEF3Record(
                    time=int(true_time),
                    record_type=record_type,
                    label=label,
                    details=details,
                    duration=duration,
                    channel=channel,
                    times={name: _records_true_times(value, clock) for name, value in body_times.items()},
                )
            )
    return records, encrypted, records_from_start


def _skipped_channel(directory: Path, passwords: MEF3Passwords | None, session_name: str) -> dict | None:
    """
    What a skipped channel adds to its session without its samples being read: its rate, the true start time
    of its first block with samples, the true end of its last one, and its recording time offset. Read from its
    first metadata file and its indices only.

    None, with a warning, when they cannot be read: a channel may be skipped because it is unreadable.
    """
    try:
        segment_directories, empty_segments = _data_segment_directories(directory)
        # A channel whose segments are all empty, one skipped because it holds no samples, still has metadata
        # and records; its first segment's metadata file gives its rate and recording time offset.
        metadata_directories = segment_directories or empty_segments
        if not metadata_directories:
            raise ValueError("it holds no .segd segment directory")
        metadata = _read_segment_metadata(_only_file(metadata_directories[0], ".tmet"), passwords)
        # As _read_channel checks it: meflib's "no entry" is -1.0, and a dead channel is one a user skips.
        if not (np.isfinite(metadata["sampling_frequency"]) and metadata["sampling_frequency"] > 0):
            raise ValueError(f"it gives a sampling frequency of {metadata['sampling_frequency']}")
        first_block_start = last_block_end = None
        for segment_directory in segment_directories:
            index = _read_time_series_index(_only_file(segment_directory, ".tidx"))
            with_samples = index[index["number_of_samples"] > 0]
            if not len(with_samples):
                continue
            starts = _true_time(with_samples["start_time"].astype(np.int64), metadata["recording_time_offset"])
            ends = starts + np.round(with_samples["number_of_samples"] * 1e6 / metadata["sampling_frequency"])
            if first_block_start is None:
                first_block_start = int(starts[0])
            segment_end = int(ends.max())
            last_block_end = segment_end if last_block_end is None else max(last_block_end, segment_end)
    except (ValueError, OSError) as error:
        warnings.warn(
            f"The skipped MEF 3 channel '{directory.name[: -len('.timd')]}' of '{session_name}' cannot be read "
            f"({error}), so its start and records are not taken into account.",
            UserWarning,
            stacklevel=_stacklevel_outside_neuroconv(),
        )
        return None
    return dict(
        sampling_frequency=metadata["sampling_frequency"],
        first_block_start=first_block_start,
        last_block_end=last_block_end,
        recording_time_offset=metadata["recording_time_offset"],
    )


def read_mef3_session(
    folder_path, channels_to_skip: list | None = None, passwords: MEF3Passwords | None = None
) -> MEF3Session:
    """
    Read the metadata and block indices of a MEF 3.0 session, without decoding any samples.

    Parameters
    ----------
    folder_path : str or Path
        The session's ``.mefd`` directory.
    channels_to_skip : list of str, optional
        Names of channels to leave out. Their samples and data files are not read. Their first metadata file
        and their indices are still read, and their records kept, so the
        session's start time and records are the same whichever channels are skipped: readers of one session
        over complementary channels_to_skip (one per sampling rate, say) share one timeline and one set of
        records.
    passwords : MEF3Passwords, optional
        The session's password, needed when its metadata is encrypted, and ignored when it is not.

    Returns
    -------
    MEF3Session
        Channels are in acquisition channel order when every channel gives a distinct one, else by name. They
        are grouped by sampling rate and runs into :attr:`MEF3Session.channel_groups`, each of which one
        ElectricalSeries can hold; a session recorded at several rates, or whose channels were recorded over
        different stretches of time, has several.
    """
    path = Path(folder_path)
    if not path.is_dir():
        raise ValueError(f"'{path}' is not a directory. A MEF 3 session is a .mefd directory.")
    channel_directories = sorted(directory for directory in path.glob("*.timd") if directory.is_dir())
    if not channel_directories:
        raise ValueError(f"'{path}' holds no .timd channel directories, so it is not a MEF 3 session.")
    names = [directory.name[: -len(".timd")] for directory in channel_directories]
    to_skip = set(channels_to_skip or [])
    unknown = sorted(to_skip - set(names))
    if unknown:
        raise ValueError(f"channels_to_skip names channels that are not in the session: {unknown}. Available: {names}.")
    skipped_directories = [directory for directory, name in zip(channel_directories, names) if name in to_skip]
    channel_directories = [directory for directory, name in zip(channel_directories, names) if name not in to_skip]
    if not channel_directories:
        raise ValueError("channels_to_skip leaves no channels in the session.")

    channels, recording_time_offsets, gmt_offsets = [], [], []
    for directory in channel_directories:
        channel, recording_time_offset, gmt_offset = _read_channel(directory, passwords)
        if channel.number_of_samples == 0:
            raise ValueError(
                f"The MEF 3 channel '{channel.name}' of '{path.name}' holds no samples. {_LEAVE_EMPTY_CHANNELS_OUT}"
            )
        channels.append(channel)
        recording_time_offsets.append(recording_time_offset)
        gmt_offsets.append(gmt_offset)

    order = _channel_order(
        [channel.name for channel in channels], [channel.acquisition_channel_number for channel in channels]
    )
    channels = [channels[index] for index in order]
    recording_time_offsets = [recording_time_offsets[index] for index in order]
    gmt_offsets = [gmt_offsets[index] for index in order]
    channel_groups = _group_channels(channels, path)

    skipped = {
        directory: summary
        for directory in skipped_directories
        if (summary := _skipped_channel(directory, passwords, path.name)) is not None
    }
    start_time = min(
        [group.runs[0].start_time for group in channel_groups]
        + [summary["first_block_start"] for summary in skipped.values() if summary["first_block_start"] is not None]
    )
    session_sampling_frequencies = tuple(
        sorted(
            {
                *(group.sampling_frequency for group in channel_groups),
                *(summary["sampling_frequency"] for summary in skipped.values()),
            }
        )
    )

    # The session-level records use the recording time offset the channels share; MEF 3 writers give every
    # segment the session's.
    session_recording_time_offset = recording_time_offsets[0]
    # The span's end, like its start, is taken over every channel, kept or skipped, so that complementary
    # channels_to_skip read the same records (see _records_clock).
    session_span = (
        start_time,
        max(
            [
                run.start_time + round(run.number_of_samples * 1e6 / group.sampling_frequency)
                for group in channel_groups
                for run in group.runs
            ]
            + [summary["last_block_end"] for summary in skipped.values() if summary["last_block_end"] is not None]
        ),
    )
    records, encrypted, records_from_start = _read_records(
        path, channel="", recording_time_offset=session_recording_time_offset, session_span=session_span
    )
    record_sources = [
        (channel.directory, channel.name, recording_time_offset)
        for channel, recording_time_offset in zip(channels, recording_time_offsets)
    ] + [
        (directory, directory.name[: -len(".timd")], summary["recording_time_offset"])
        for directory, summary in skipped.items()
    ]
    for directory, name, recording_time_offset in record_sources:
        for level_directory in [directory, *_segment_directories(directory)]:
            level_records, level_encrypted, level_from_start = _read_records(
                level_directory, name, recording_time_offset, session_span
            )
            records += level_records
            encrypted += level_encrypted
            records_from_start += level_from_start
    if records_from_start:
        warnings.warn(
            f"{records_from_start} records of the MEF 3 session '{path.name}' are stored offset by the session's start "
            "rather than by its recording time offset: read with that offset none of them falls within the "
            "session, and read from its start every one does. They are placed from the session's start.",
            UserWarning,
            stacklevel=_stacklevel_outside_neuroconv(),
        )
    if encrypted:
        warnings.warn(
            f"{encrypted} records of the MEF 3 session '{path.name}' are encrypted and are left out.",
            UserWarning,
            stacklevel=_stacklevel_outside_neuroconv(),
        )
    # Time first; the rest only orders records at the same instant the same way whichever channels are skipped.
    records.sort(key=lambda record: (record.time, record.channel, record.record_type, record.label, record.details))

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
        channel_groups=channel_groups,
        gmt_offset=gmt_offsets[0],
        records=records,
        lossy_blocks=lossy_blocks,
        largest_scale_factor=largest_scale_factor,
        start_time=start_time,
        session_sampling_frequencies=session_sampling_frequencies,
    )


def _channel_order(names: list[str], acquisition_channel_numbers: list[int]) -> list[int]:
    """Positions in acquisition channel order when every channel gives a distinct number, else in name order."""
    numbers = list(acquisition_channel_numbers)
    if all(number >= 0 for number in numbers) and len(set(numbers)) == len(numbers):
        return [int(index) for index in np.argsort(numbers, kind="stable")]
    return sorted(range(len(names)), key=lambda index: names[index])


def channel_sampling_frequencies(folder_path, passwords: MEF3Passwords | None = None) -> dict[str, float]:
    """
    Each time series channel's sampling frequency, in Hz, in the order :func:`read_mef3_session` gives them.

    Only the first metadata file of each channel is read, of its first segment that holds data; ``passwords``
    as for :func:`read_mef3_session`. This says which channels are sampled at which rate without reading any
    index, to choose channels to leave out with ``channels_to_skip``.
    """
    path = Path(folder_path)
    channel_directories = sorted(directory for directory in path.glob("*.timd") if directory.is_dir())
    names, numbers, rates = [], [], []
    for directory in channel_directories:
        segment_directories = _data_segment_directories(directory)[0] or _segment_directories(directory)
        if not segment_directories:
            raise ValueError(f"The MEF 3 channel directory '{directory}' holds no .segd segment directories.")
        metadata = _read_segment_metadata(_only_file(segment_directories[0], ".tmet"), passwords)
        names.append(directory.name[: -len(".timd")])
        numbers.append(metadata["acquisition_channel_number"])
        rates.append(metadata["sampling_frequency"])
    return {names[index]: rates[index] for index in _channel_order(names, numbers)}


def _holds_no_samples(channel_directory: Path) -> bool:
    """
    Whether a channel has segments but none of them holds a sample: each is empty (see :func:`_is_empty_segment`),
    or its index lists no block with samples.

    Read from the indices alone, up to the first that lists a block with samples. A channel with no segment
    directory is not said to be empty, and an index read on the way that cannot be read raises: neither is a
    channel that holds no samples, and the reader refuses both as they are.
    """
    if not _segment_directories(channel_directory):
        return False
    segment_directories, _ = _data_segment_directories(channel_directory)
    return not any(
        (_read_time_series_index(_only_file(segment_directory, ".tidx"))["number_of_samples"] > 0).any()
        for segment_directory in segment_directories
    )


def empty_channel_names(folder_path) -> list[str]:
    """
    The names of a session's time series channels that hold no samples, in name order.

    A writer may create a channel and never write a block with samples to it. :func:`read_mef3_session` refuses
    such a channel unless it is in ``channels_to_skip``; this lists them, to leave them out. Only the block
    indices are read, which MEF 3 never encrypts, so no password is needed. A channel whose index cannot be read
    is not listed: reading its index raises.
    """
    path = Path(folder_path)
    if not path.is_dir():
        raise ValueError(f"'{path}' is not a directory. A MEF 3 session is a .mefd directory.")
    channel_directories = sorted(directory for directory in path.glob("*.timd") if directory.is_dir())
    return [directory.name[: -len(".timd")] for directory in channel_directories if _holds_no_samples(directory)]


def available_channel_names(folder_path, passwords: MEF3Passwords | None = None) -> list[str]:
    """
    The names of a session's time series channels, in the order :func:`read_mef3_session` gives them.

    Only the first segment's metadata file of each channel is read; ``passwords`` as for :func:`read_mef3_session`.
    """
    return list(channel_sampling_frequencies(folder_path, passwords))
