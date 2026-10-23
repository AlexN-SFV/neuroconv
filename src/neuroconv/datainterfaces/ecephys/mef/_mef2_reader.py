"""
Reader for MEF 2 recordings (Multiscale Electrophysiology Format 2.0 and 2.1, Mayo Systems Electrophysiology Lab).

A MEF 2 recording is a directory of ``.mef`` files, one per channel. Each file holds:

* a 1024-byte header of fields at fixed offsets;
* the samples from byte 1024, as RED-compressed blocks (see :mod:`._red`), each with a 287-byte header;
* the block index, after the blocks: one 24-byte entry per block giving its start time, its offset in the
  file and the number of its first sample;
* in MEF 2.1, optionally, the discontinuity index after that: the 1-based numbers of the blocks that follow
  a discontinuity.

Every offset here is that of ``mef.h`` and ``mef_lib.c`` in Mayo's mef_lib 2.1, the format's reference
implementation, in C; no Python MEF 2 reader exists to check against. MEF 2.0 shares the 2.1 layout up to
the GMT offset at byte 836, and has neither the discontinuity index nor a header CRC. Its blocks carry the
same discontinuity flag, which is what the runs are built from in both versions.

Only what is needed to name, place and scale the samples is read. In particular the subject fields (names,
ID and password, bytes 176-351), the anonymized subject name, the institution and the free-text comment
fields are never read, so they cannot reach the NWB file.

Every read is a byte range through a file object: a header, an index, or the blocks a request needs. Nothing
loads a whole data file, which can run to several GB per channel.
"""

import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from ._mef3_reader import MEF3Run, _channel_order, group_blocks_into_runs
from ._red import MEF2_RED_LAYOUT, decode_red_blocks, koopman_crc32
from .._channel_ids import (
    _stacklevel_outside_neuroconv,
    indices_to_keep,
    unique_channel_ids,
    unknown_channels_to_skip,
)

HEADER_BYTES = 1024  # MEF_HEADER_LENGTH
HEADER_CRC_OFFSET = 1020
INDEX_ENTRY_BYTES = 24  # sizeof(INDEX_DATA)
DISCONTINUITY_ENTRY_BYTES = 8
BLOCK_HEADER_BYTES = 287  # BLOCK_HEADER_BYTES
BLOCK_FIXED_HEADER_BYTES = 31  # the fields before RED_STAT_MODEL_OFFSET
MEF_LITTLE_ENDIAN = 1
# RED_DISCONTINUITY_OFFSET holds the block flags; bit 0 marks a block that follows a discontinuity.
_DISCONTINUITY_FLAG_OFFSET = 30
_DISCONTINUITY_MASK = 0x01

# The fields of an index entry (INDEX_DATA).
_INDEX_DTYPE = np.dtype([("time", "<u8"), ("file_offset", "<u8"), ("sample_number", "<u8")])
# The fixed fields of a RED block header (RED_*_OFFSET); the maximum and minimum are si3s.
_BLOCK_HEADER_DTYPE = np.dtype(
    {
        "names": [
            "crc",
            "compressed_bytes",
            "start_time",
            "difference_count",
            "number_of_samples",
            "maximum",
            "minimum",
            "flags",
        ],
        "formats": ["<u4", "<u4", "<u8", "<u4", "<u4", ("u1", 3), ("u1", 3), "u1"],
        "offsets": [0, 4, 8, 16, 20, 24, 27, 30],
        "itemsize": BLOCK_FIXED_HEADER_BYTES,
    }
)


def _si3(raw: np.ndarray) -> np.ndarray:
    """Little-endian si3 fields, as int64."""
    raw = np.atleast_2d(raw).astype(np.int64)
    value = raw[:, 0] | (raw[:, 1] << 8) | (raw[:, 2] << 16)
    return np.where(value >= 1 << 23, value - (1 << 24), value)


@dataclass
class MEF2Channel:
    """One ``.mef`` file: the header fields the conversion uses and the index of every block holding samples."""

    name: str
    path: Path
    version: str
    sampling_frequency: float
    voltage_conversion_factor: float  # µV per count; negative when the counts are inverted
    physical_channel_number: int
    acquisition_system: str
    gmt_offset: float | None  # hours east of UTC
    # One entry per block with samples, in file order: where it starts in the file, how many bytes there are
    # before the next block (its length plus any alignment padding), its stated start time in µUTC, its
    # number of samples and whether it follows a discontinuity.
    blocks: np.ndarray = field(repr=False)
    # Whether a one-sample placeholder block before the first real one was left out.
    dropped_placeholder_block: bool = False
    # Whether read_blocks checks each block's CRC.
    validate_crc: bool = False
    runs: list[MEF3Run] = field(default_factory=list, repr=False)

    @property
    def number_of_samples(self) -> int:
        return int(self.blocks["number_of_samples"].sum())

    def read_blocks(self, first_block: int, number_of_blocks: int) -> np.ndarray:
        """Decode ``number_of_blocks`` consecutive blocks into one int32 array, from one contiguous read."""
        blocks = self.blocks[first_block : first_block + number_of_blocks]
        first_byte = int(blocks["file_offset"][0])
        last_byte = int(blocks["file_offset"][-1] + blocks["span"][-1])
        with open(self.path, "rb") as file:
            file.seek(first_byte)
            buffer = np.frombuffer(file.read(last_byte - first_byte), dtype=np.uint8)
        if buffer.size != last_byte - first_byte:
            raise ValueError(f"The MEF 2 file '{self.path}' ends before the blocks its index lists.")

        starts = blocks["file_offset"].astype(np.int64) - first_byte
        headers = buffer[starts[:, None] + np.arange(BLOCK_FIXED_HEADER_BYTES)].copy().view(_BLOCK_HEADER_DTYPE)[:, 0]
        block_bytes = BLOCK_HEADER_BYTES + headers["compressed_bytes"].astype(np.int64)
        mismatched = np.flatnonzero(
            (headers["number_of_samples"] != blocks["number_of_samples"])
            | (headers["start_time"].astype(np.int64) != blocks["start_time"])
            | (block_bytes > blocks["span"])
        )
        if mismatched.size:
            block = first_block + int(mismatched[0])
            raise ValueError(
                f"Block {block} of the MEF 2 file '{self.path}' does not have the start time, number of samples or "
                f"size its index gives it ({mismatched.size} such blocks among those read). The file is corrupt."
            )
        if self.validate_crc:
            crcs = koopman_crc32(buffer, starts + 4, starts + block_bytes)
            bad = np.flatnonzero(crcs != headers["crc"])
            if bad.size:
                raise ValueError(
                    f"Block {first_block + int(bad[0])} of the MEF 2 file '{self.path}' fails its CRC "
                    f"({bad.size} such blocks among those read). The file is corrupt."
                )
        ones = np.ones(len(blocks))
        zeros = np.zeros(len(blocks))
        return decode_red_blocks(
            buffer,
            block_starts=starts,
            block_bytes=block_bytes,
            # RED_decompress_block decodes one symbol more than the header counts: the keysample flag of the
            # first sample, which the encoder leaves implied.
            difference_counts=headers["difference_count"].astype(np.int64) + 1,
            sample_counts=blocks["number_of_samples"],
            scale_factors=ones,
            detrend_slopes=zeros,
            detrend_intercepts=zeros,
            layout=MEF2_RED_LAYOUT,
        )


@dataclass
class MEF2Annotation:
    """One annotation of an IEEG.org ``annotations.xml``."""

    start_time: int  # µUTC
    end_time: int  # µUTC
    annotation_type: str
    description: str
    layer: str
    annotator: str
    channels: list[str]  # the .mef file names the annotation applies to, without the suffix


@dataclass
class MEF2Recording:
    """The channels of a MEF 2 recording directory at one sampling rate, and the runs they share."""

    path: Path
    channels: list[MEF2Channel]
    channel_ids: list[str]  # the channels' names, unique within the stream
    channel_labels: list[str]  # the names the files give them, which may repeat
    runs: list[MEF3Run]
    sampling_frequency: float
    gmt_offset: float | None
    # µUTC, the earliest start time the directory's headers state, whichever stream their files are in. It is
    # what IEEG.org annotation times count from.
    recording_start_time: int
    stream_name: str
    number_of_streams: int  # sampling rates among the directory's files, skipped ones included

    @property
    def start_time(self) -> int:
        """
        The origin of the recording's timeline, in µUTC: the header start time, or the first run's if earlier.

        Not the first run's start, so that every stream of the directory shares one origin, and NWB times
        equal IEEG.org's offsets. In PennEPI's files the first run starts 2.5 s after it, the header start
        being the time of the placeholder block that is left out.
        """
        return min(self.recording_start_time, self.runs[0].start_time)

    @property
    def start_datetime(self) -> datetime:
        """:attr:`start_time` as a datetime, in the recording's own time zone when its header gives one."""
        start = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(microseconds=self.start_time)
        if self.gmt_offset is not None:
            start = start.astimezone(timezone(timedelta(hours=self.gmt_offset)))
        return start

    def locate(self, time: int) -> tuple[int, float]:
        """
        Where a µUTC time falls: the index of its run, and how many seconds into that run.

        Each run is written at the nominal sampling rate from its first block's time, while the block times
        within it follow the recording's clock, which drifts against that rate. A time is therefore placed
        on the sample the block index puts it on, which is where a viewer that reads the block times, such
        as IEEG.org's, showed it. Where two blocks claim the time, after the clock was set back, the later
        one is used; a time in a gap is placed from the start of the run before it.
        """
        blocks = self.channels[0].blocks
        start_times = blocks["start_time"]
        counts = blocks["number_of_samples"]
        ends = start_times + counts * (1e6 / self.sampling_frequency)
        covering = np.flatnonzero((start_times <= time) & (time < ends))
        if covering.size:
            block = int(covering[-1])
            run_firsts = np.array([run.first_block for run in self.runs])
            run_index = int(np.searchsorted(run_firsts, block, side="right") - 1)
            samples = int(counts[self.runs[run_index].first_block : block].sum())
            return run_index, samples / self.sampling_frequency + (time - int(start_times[block])) / 1e6
        run_index = 0
        for index, run in enumerate(self.runs):
            if time >= run.start_time:
                run_index = index
        return run_index, (time - self.runs[run_index].start_time) / 1e6


def _text(raw: bytes) -> str:
    """A NUL-terminated field. MEF 2 predates any declared encoding, so Latin-1, which cannot fail."""
    return raw.split(b"\0", 1)[0].decode("latin-1").strip()


def _number(raw: bytes, offset: int, dtype: str):
    return np.frombuffer(raw, dtype=dtype, count=1, offset=offset)[0].item()


def read_mef2_header(path, validate_crc: bool = False) -> dict:
    """
    The fields of a ``.mef`` file's header that the conversion uses, refusing what it cannot read.

    Parameters
    ----------
    path : str or Path
    validate_crc : bool, default: False
        Raise, rather than warn, when a MEF 2.1 header fails its CRC. MEF 2.0 headers have none.
    """
    path = Path(path)
    with open(path, "rb") as file:
        raw = file.read(HEADER_BYTES)
    if len(raw) < HEADER_BYTES:
        raise ValueError(f"'{path}' is shorter than a MEF 2 header, so it is not a MEF 2 file.")
    major, minor, byte_order = raw[164], raw[165], raw[163]
    if major != 2:
        raise ValueError(f"'{path}' does not have a MEF 2 header (it states major version {major}).")
    if minor not in (0, 1):
        raise ValueError(
            f"'{path}' is a MEF 2.{minor} file. Only MEF 2.0 and 2.1 are supported; MEF 2.2 can shift every "
            "timestamp by an offset kept in the encrypted part of the header."
        )
    if byte_order != MEF_LITTLE_ENDIAN:
        raise ValueError(f"'{path}' is a big-endian MEF 2 file, which is not supported.")
    # SUBJECT_ENCRYPTION_USED, SESSION_ENCRYPTION_USED, DATA_ENCRYPTION_USED. Subject encryption covers only
    # the subject fields, which are never read.
    if raw[161] or raw[162]:
        raise ValueError(
            f"The MEF 2 file '{path}' is encrypted (session encryption {raw[161]}, data encryption {raw[162]}). "
            "Encrypted MEF 2 files are not supported; decrypt them with the MEF tools and their passwords first."
        )
    compression = _text(raw[756:788])
    if "RED" not in compression.upper() and "RANGE ENCODED" not in compression.upper():
        raise ValueError(f"The MEF 2 file '{path}' is compressed with {compression!r}, not RED, so it cannot be read.")
    block_header_length = _number(raw, 832, "<u2")
    if block_header_length != BLOCK_HEADER_BYTES:
        raise ValueError(
            f"The MEF 2 file '{path}' states {block_header_length}-byte block headers, not the {BLOCK_HEADER_BYTES} "
            "bytes of RED, so it cannot be read."
        )

    version = f"{major}.{minor}"
    if minor >= 1:
        stored = _number(raw, HEADER_CRC_OFFSET, "<u4")
        computed = int(
            koopman_crc32(np.frombuffer(raw, dtype=np.uint8), np.array([0]), np.array([HEADER_CRC_OFFSET]))[0]
        )
        if stored != computed:
            message = f"The header of the MEF 2 file '{path}' fails its CRC, so its fields may be corrupt."
            if validate_crc:
                raise ValueError(message)
            warnings.warn(message, UserWarning, stacklevel=_stacklevel_outside_neuroconv())

    gmt_offset = _number(raw, 836, "<f4") if minor >= 1 else None
    if gmt_offset is not None and not (np.isfinite(gmt_offset) and abs(gmt_offset) <= 24):
        gmt_offset = None
    return dict(
        path=path,
        version=version,
        number_of_samples=_number(raw, 368, "<u8"),
        channel_name=_text(raw[376:408]) or path.stem,
        recording_start_time=_number(raw, 408, "<u8"),
        sampling_frequency=_number(raw, 424, "<f8"),
        voltage_conversion_factor=_number(raw, 456, "<f8"),
        acquisition_system=_text(raw[464:496]),
        physical_channel_number=_number(raw, 752, "<i4"),
        index_data_offset=_number(raw, 816, "<u8"),
        number_of_index_entries=_number(raw, 824, "<u8"),
        gmt_offset=gmt_offset,
        discontinuity_data_offset=_number(raw, 840, "<u8") if minor >= 1 else 0,
        number_of_discontinuity_entries=_number(raw, 848, "<u8") if minor >= 1 else 0,
    )


def _read_range(file, offset: int, number_of_bytes: int, path: Path, what: str) -> bytes:
    file.seek(offset)
    raw = file.read(number_of_bytes)
    if len(raw) < number_of_bytes:
        raise ValueError(f"The MEF 2 file '{path}' ends before its {what}.")
    return raw


def _read_blocks(header: dict) -> tuple[np.ndarray, bool]:
    """The blocks of one file with samples, and whether a placeholder first block was left out."""
    path = header["path"]
    number_of_blocks = header["number_of_index_entries"]
    index_offset = header["index_data_offset"]
    if number_of_blocks == 0:
        raise ValueError(f"The MEF 2 file '{path}' has an empty block index.")
    with open(path, "rb") as file:
        index = np.frombuffer(
            _read_range(file, index_offset, number_of_blocks * INDEX_ENTRY_BYTES, path, "block index"),
            dtype=_INDEX_DTYPE,
        )
        file_offsets = index["file_offset"].astype(np.int64)
        sample_numbers = index["sample_number"].astype(np.int64)
        counts = np.diff(np.append(sample_numbers, header["number_of_samples"]))
        spans = np.diff(np.append(file_offsets, index_offset))
        if (
            file_offsets[0] < HEADER_BYTES
            or (spans < BLOCK_FIXED_HEADER_BYTES).any()
            or sample_numbers[0] != 0
            or (counts < 0).any()
        ):
            raise ValueError(
                f"The block index of the MEF 2 file '{path}' is not in file order or does not add up to the "
                f"{header['number_of_samples']} samples its header states. The file is corrupt."
            )

        flags = _discontinuity_flags_from_index(file, header)
        if flags is None:
            # MEF 2.0, or a 2.1 file whose writer left the discontinuity index out (mef_lib's own write_mef
            # does): every block header carries the flag.
            flags = np.empty(number_of_blocks, dtype=bool)
            for position, offset in enumerate(file_offsets):
                file.seek(int(offset) + _DISCONTINUITY_FLAG_OFFSET)
                flags[position] = bool(file.read(1)[0] & _DISCONTINUITY_MASK)
        first_header = np.frombuffer(
            _read_range(file, int(file_offsets[0]), BLOCK_FIXED_HEADER_BYTES, path, "first block"),
            dtype=_BLOCK_HEADER_DTYPE,
        )[0]

    blocks = np.empty(
        number_of_blocks,
        dtype=[
            ("file_offset", "i8"),
            ("span", "i8"),
            ("start_time", "i8"),
            ("number_of_samples", "i8"),
            ("discontinuity", "?"),
        ],
    )
    blocks["file_offset"] = file_offsets
    blocks["span"] = spans
    blocks["start_time"] = index["time"].astype(np.int64)
    blocks["number_of_samples"] = counts
    blocks["discontinuity"] = flags

    # IEEG.org's MEF 2.1 writer opens every file with a block of one sample of value 0, stamped seconds before
    # the recording proper, which is flagged as following it. It is a placeholder, not a sample.
    dropped = False
    if (
        number_of_blocks > 1
        and counts[0] == 1
        and _si3(first_header["maximum"])[0] == 0 == _si3(first_header["minimum"])[0]
    ):
        elapsed_samples = np.rint(
            (blocks["start_time"][1] - blocks["start_time"][0]) * header["sampling_frequency"] / 1e6
        )
        if elapsed_samples != 1:
            blocks = blocks[1:]
            blocks["discontinuity"][0] = True
            dropped = True
    # A block without samples is left out, and its flag passed to the block after it: a writer can mark a
    # dropout with an empty flagged block, and the samples after it must not be taken to continue the run.
    for position in np.flatnonzero(blocks["number_of_samples"] == 0):
        if blocks["discontinuity"][position] and position + 1 < len(blocks):
            blocks["discontinuity"][position + 1] = True
    return blocks[blocks["number_of_samples"] > 0], dropped


def _discontinuity_flags_from_index(file, header: dict) -> np.ndarray | None:
    """
    Each block's discontinuity flag from a MEF 2.1 discontinuity index, or None where there is none to use.

    The index holds block numbers counted from 1: in PennEPI's files every entry less one is a block whose
    start time leaves a gap after the samples before it, and the first two entries are the first two blocks.
    """
    count = header["number_of_discontinuity_entries"]
    offset = header["discontinuity_data_offset"]
    number_of_blocks = header["number_of_index_entries"]
    if count == 0 or offset == 0:
        return None
    raw = _read_range(file, offset, count * DISCONTINUITY_ENTRY_BYTES, header["path"], "discontinuity index")
    entries = np.frombuffer(raw, dtype="<u8").astype(np.int64)
    if entries.min() < 1 or entries.max() > number_of_blocks:
        # Not the 1-based block numbers it should hold; the block headers are authoritative.
        return None
    flags = np.zeros(number_of_blocks, dtype=bool)
    flags[entries - 1] = True
    return flags


def _mef_files(path: Path) -> list[Path]:
    return sorted(child for child in path.iterdir() if child.is_file() and child.suffix.lower() == ".mef")


def _stream_name(sampling_frequency: float) -> str:
    return f"{sampling_frequency:g} Hz"


def _identify(path: Path) -> dict:
    """
    The name, rate and physical channel number a ``.mef`` file states, read without refusing anything.

    Enough to order and name the channels and to match ``channels_to_skip``, so a file that cannot be
    converted can still be left out. A file without a readable MEF 2 header, or whose session fields are
    encrypted, is named by its file name.
    """
    with open(path, "rb") as file:
        raw = file.read(HEADER_BYTES)
    if len(raw) < HEADER_BYTES or raw[164] != 2 or raw[161]:
        return dict(
            path=path,
            channel_name=path.stem,
            sampling_frequency=float("nan"),
            physical_channel_number=-1,
            recording_start_time=0,
        )
    return dict(
        path=path,
        channel_name=_text(raw[376:408]) or path.stem,
        sampling_frequency=_number(raw, 424, "<f8"),
        physical_channel_number=_number(raw, 752, "<i4"),
        # 0 is the header's "no entry".
        recording_start_time=_number(raw, 408, "<u8"),
    )


def _identify_files(path: Path) -> list[dict]:
    """:func:`_identify` for each of a directory's .mef files, in channel order (see :func:`_channel_order`)."""
    if not path.is_dir():
        raise ValueError(f"'{path}' is not a directory. A MEF 2 recording is a directory of .mef files.")
    files = _mef_files(path)
    if not files:
        raise ValueError(f"'{path}' holds no .mef files, so it is not a MEF 2 recording.")
    identities = [_identify(file) for file in files]
    order = _channel_order(
        [identity["channel_name"] for identity in identities],
        [identity["physical_channel_number"] for identity in identities],
    )
    return [identities[index] for index in order]


def _channel_ids(identities: list[dict]) -> list[str]:
    """Unique ids per stream, so a label recurring at another rate keeps its name, as EDF's streams do."""
    return unique_channel_ids(
        [identity["channel_name"] for identity in identities],
        groups=[identity["sampling_frequency"] for identity in identities],
    )


def stream_names(folder_path) -> list[str]:
    """The streams of a MEF 2 recording: one per sampling rate, named like ``"1024 Hz"``, slowest first."""
    rates = {identity["sampling_frequency"] for identity in _identify_files(Path(folder_path))}
    return [_stream_name(rate) for rate in sorted(rate for rate in rates if np.isfinite(rate) and rate > 0)]


def available_channel_names(folder_path) -> list[str]:
    """The channel ids of every ``.mef`` file in a directory, in the order :func:`read_mef2_recording` gives them."""
    return _channel_ids(_identify_files(Path(folder_path)))


def _shown_blocks(channel: MEF2Channel, indices: np.ndarray, sampling_frequency: float) -> str:
    blocks = channel.blocks
    jumps = [
        (blocks["start_time"][index] - blocks["start_time"][index - 1]) / 1e6
        - blocks["number_of_samples"][index - 1] / sampling_frequency
        for index in indices[:3]
    ]
    return ", ".join(f"block {index} ({jump:+.3f} s)" for index, jump in zip(indices, jumps))


def _check_shared_layout(channels: list[MEF2Channel], sampling_frequency: float, path: Path) -> list[MEF3Run]:
    """The runs of the recording, which every channel must share: one electrodes table spans every series."""
    reference = channels[0]
    runs, unflagged_jumps = group_blocks_into_runs(reference.blocks, sampling_frequency, backward_jumps_break=False)
    reference.runs = runs
    if unflagged_jumps.size:
        blocks = reference.blocks
        elapsed = blocks["start_time"][unflagged_jumps] - blocks["start_time"][unflagged_jumps - 1]
        expected = blocks["number_of_samples"][unflagged_jumps - 1] * (1e6 / sampling_frequency)
        forward, backward = unflagged_jumps[elapsed > expected], unflagged_jumps[elapsed < expected]
        if forward.size:
            warnings.warn(
                f"The timestamps of the MEF 2 channel '{reference.name}' of '{path.name}' jump forward by more than "
                f"a second at {forward.size} blocks that are not flagged as discontinuities: "
                f"{_shown_blocks(reference, forward, sampling_frequency)}. Each is taken as a gap and starts a new "
                "ElectricalSeries.",
                UserWarning,
                stacklevel=_stacklevel_outside_neuroconv(),
            )
        if backward.size:
            warnings.warn(
                f"The timestamps of the MEF 2 channel '{reference.name}' of '{path.name}' go back by more than a "
                f"second at {backward.size} blocks that are not flagged as discontinuities: "
                f"{_shown_blocks(reference, backward, sampling_frequency)}. The writer set its clock back while "
                "the samples ran on, so the run continues across them, at the nominal sampling rate; samples "
                "after such a block are placed up to that much later than its timestamp.",
                UserWarning,
                stacklevel=_stacklevel_outside_neuroconv(),
            )
    overlapping = [
        index
        for index in range(1, len(runs))
        if runs[index].start_time
        < runs[index - 1].start_time + runs[index - 1].number_of_samples * 1e6 / sampling_frequency
    ]
    if overlapping:
        warnings.warn(
            f"{len(overlapping)} runs of the MEF 2 recording '{path.name}' start before the run before them ends "
            f"(runs {overlapping[:5]}), at blocks flagged as discontinuities, so their ElectricalSeries overlap in "
            "time.",
            UserWarning,
            stacklevel=_stacklevel_outside_neuroconv(),
        )

    half_sample = 0.5e6 / sampling_frequency
    for channel in channels[1:]:
        channel_runs, _ = group_blocks_into_runs(channel.blocks, sampling_frequency, backward_jumps_break=False)
        channel.runs = channel_runs
        same = len(channel_runs) == len(runs) and all(
            ours.number_of_samples == theirs.number_of_samples
            and abs(ours.start_time - theirs.start_time) < half_sample
            for ours, theirs in zip(runs, channel_runs)
        )
        if not same:
            raise ValueError(
                f"The MEF 2 channels '{reference.name}' and '{channel.name}' of '{path.name}' were not recorded over "
                f"the same stretches of time ({len(runs)} runs of {[run.number_of_samples for run in runs][:5]} "
                f"samples against {len(channel_runs)} runs of {[run.number_of_samples for run in channel_runs][:5]}). "
                "Convert them separately, leaving the others out with channels_to_skip."
            )
    return runs


def read_mef2_recording(
    folder_path,
    channels_to_skip: list | None = None,
    stream_name: str | None = None,
    validate_crc: bool = False,
) -> MEF2Recording:
    """
    Read the headers and block indices of a MEF 2 recording directory, without decoding any samples.

    Parameters
    ----------
    folder_path : str or Path
        The directory holding the recording's ``.mef`` files, one per channel.
    channels_to_skip : list of str, optional
        Channels to leave out, by id or by the name the file gives them. Only enough of their headers is read
        to name them, so a file that cannot be converted, such as an encrypted one, can be left out.
    stream_name : str, optional
        The sampling rate to read, as :func:`stream_names` names it. Required when the files are sampled at
        more than one rate.
    validate_crc : bool, default: False
        Check every block's CRC as it is decoded, and raise rather than warn on a MEF 2.1 header that fails
        its CRC.

    Returns
    -------
    MEF2Recording
        Channels are in physical channel order when every channel gives a distinct one, else by name.
    """
    path = Path(folder_path)
    identities = _identify_files(path)
    labels = [identity["channel_name"] for identity in identities]
    ids = _channel_ids(identities)
    unknown = unknown_channels_to_skip(labels, ids, channels_to_skip)
    if unknown:
        raise ValueError(f"channels_to_skip names channels that are not in the recording: {unknown}. Available: {ids}.")
    keep = indices_to_keep(labels, ids, channels_to_skip)
    if not keep:
        raise ValueError("channels_to_skip leaves no channels in the recording.")

    headers = {index: read_mef2_header(identities[index]["path"], validate_crc=validate_crc) for index in keep}
    streams = {}
    for index in keep:
        rate = headers[index]["sampling_frequency"]
        if not (np.isfinite(rate) and rate > 0):
            raise ValueError(f"The MEF 2 file '{headers[index]['path']}' gives a sampling frequency of {rate}.")
        streams.setdefault(_stream_name(rate), []).append(index)
    if stream_name is None:
        if len(streams) > 1:
            described = "; ".join(
                f"{name}: {', '.join(ids[index] for index in indices[:5])}" for name, indices in streams.items()
            )
            raise ValueError(
                f"The channels of the MEF 2 recording '{path.name}' are sampled at {len(streams)} different rates "
                f"({described}). An ElectricalSeries has one rate, so choose one with stream_name, one of "
                f"{sorted(streams)}, and convert each separately."
            )
        stream_name = next(iter(streams))
    elif stream_name not in streams:
        raise ValueError(
            f"stream_name {stream_name!r} is not a stream of the MEF 2 recording '{path.name}'; its kept channels "
            f"have the streams {sorted(streams)}."
        )
    keep = streams[stream_name]
    sampling_frequency = headers[keep[0]]["sampling_frequency"]

    channels = []
    for index in keep:
        header = headers[index]
        blocks, dropped = _read_blocks(header)
        if len(blocks) == 0:
            raise ValueError(
                f"The MEF 2 file '{header['path'].name}' of '{path.name}' holds no samples. Leave it out with "
                "channels_to_skip."
            )
        channels.append(
            MEF2Channel(
                name=ids[index],
                path=header["path"],
                version=header["version"],
                sampling_frequency=header["sampling_frequency"],
                voltage_conversion_factor=header["voltage_conversion_factor"],
                physical_channel_number=header["physical_channel_number"],
                acquisition_system=header["acquisition_system"],
                gmt_offset=header["gmt_offset"],
                blocks=blocks,
                dropped_placeholder_block=dropped,
                validate_crc=validate_crc,
            )
        )
    runs = _check_shared_layout(channels, sampling_frequency, path)
    # IEEG.org gives annotation times as offsets from the start of the whole recording. Taken over every file,
    # skipped ones and other streams included, so the streams of one directory share it.
    stated_starts = [
        identity["recording_start_time"] for identity in identities if identity["recording_start_time"] > 0
    ]
    rates = {identity["sampling_frequency"] for identity in identities}
    return MEF2Recording(
        path=path,
        channels=channels,
        channel_ids=[ids[index] for index in keep],
        channel_labels=[labels[index] for index in keep],
        runs=runs,
        sampling_frequency=sampling_frequency,
        gmt_offset=channels[0].gmt_offset,
        recording_start_time=min(stated_starts) if stated_starts else runs[0].start_time,
        stream_name=stream_name,
        number_of_streams=sum(1 for rate in rates if np.isfinite(rate) and rate > 0),
    )


def read_ieeg_annotations(file_path, recording_start_time: int) -> list[MEF2Annotation]:
    """
    The annotations of an IEEG.org ``annotations.xml``, in time order.

    Each ``<annotation>`` gives its start and end as ``startOffsetUsecs`` and ``endOffsetUsecs``, microseconds
    from the start of the recording, and its ``type``, ``layer`` and ``annotator``, with the ``.mef`` files it
    applies to listed under ``<channels>``.
    """
    import xml.etree.ElementTree as ElementTree

    root = ElementTree.parse(file_path).getroot()
    annotations = []
    for element in root.iter("annotation"):
        attributes = element.attrib
        try:
            start = int(attributes["startOffsetUsecs"])
            end = int(attributes.get("endOffsetUsecs", start))
        except (KeyError, ValueError):
            raise ValueError(
                f"An annotation of '{file_path}' (id {attributes.get('id')!r}) has no valid startOffsetUsecs, so "
                "it is not an IEEG.org annotations file."
            )
        channels = [
            Path(channel.attrib.get("fileName", "")).stem
            for channel in element.iter("channel")
            if channel.attrib.get("fileName")
        ]
        description = attributes.get("description") or (element.findtext("description") or "").strip()
        annotations.append(
            MEF2Annotation(
                start_time=recording_start_time + start,
                end_time=recording_start_time + max(end, start),
                annotation_type=attributes.get("type", ""),
                description=description,
                layer=attributes.get("layer", ""),
                annotator=attributes.get("annotator", ""),
                channels=channels,
            )
        )
    annotations.sort(key=lambda annotation: annotation.start_time)
    return annotations
