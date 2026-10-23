"""Tests for the MEF2RecordingInterface and its reader.

The recordings in ``mef2_fixtures/`` are excerpts of real MEF 2 files cut by ``build_mef2_fixtures.py``: their
blocks are copied byte for byte, and only the header and the block index are rewritten for the shorter file.
No Python MEF 2 reader exists to compare with, so the decoded samples are checked against what each block
header stores: its maximum and minimum value and its CRC. ``expected.json`` holds what the full source files
decode to over the same blocks.

Variations the excerpts do not have (gaps, clock steps, other channels and rates, encryption, ...) are made at
test time by :func:`write_mef2`, which lays out a MEF 2 file from a header and block bytes as mef_lib's
``write_mef_ind`` does, after :func:`patch_block` or :func:`patch_header` changed what is needed and redid the
CRCs.
"""

import hashlib
import json
import shutil
import struct
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from pynwb import NWBHDF5IO
from pynwb.testing.mock.file import mock_NWBFile

from neuroconv.datainterfaces import MEF2RecordingInterface
from neuroconv.datainterfaces.ecephys.mef._mef2_extractor import (
    MEF2RecordingExtractor,
)
from neuroconv.datainterfaces.ecephys.mef._mef2_reader import (
    read_ieeg_annotations,
    read_mef2_header,
    read_mef2_recording,
)
from neuroconv.datainterfaces.ecephys.mef._red import (
    MEF2_RED_LAYOUT,
    NUMBA_AVAILABLE,
    decode_red_blocks,
    koopman_crc32,
)
from neuroconv.tools.testing.data_interface_mixins import (
    RecordingExtractorInterfaceTestMixin,
)

FIXTURES = Path(__file__).parent / "mef2_fixtures"
# Absent only while build_mef2_fixtures.py, which imports this module, is writing it.
EXPECTED = json.loads((FIXTURES / "expected.json").read_text()) if (FIXTURES / "expected.json").exists() else {}
# MEF 2.1, PennEPI00082 (Pennsieve Discover 549, CC-BY): the placeholder block, then 8 runs split by real dropouts.
PENN = FIXTURES / "PennEPI00082_LI07_excerpt"
# MEF 2.0, Discover 13 (CC0): two runs, split by a flagged gap of 20 minutes.
AD3 = FIXTURES / "Discover13_AD3_excerpt"
# MEF 2.0, Discover 13: one run across an unflagged step of the clock back by 1.017 s.
AD3_STEP = FIXTURES / "Discover13_AD3_clock_step"

DECODERS = [
    pytest.param(True, id="numba", marks=pytest.mark.skipif(not NUMBA_AVAILABLE, reason="numba is not installed")),
    pytest.param(False, id="numpy"),
]

HEADER_BYTES = 1024
BLOCK_HEADER_BYTES = 287


def digest(samples) -> str:
    return hashlib.sha256(np.asarray(samples, dtype="<i4").tobytes()).hexdigest()


def si3(raw: bytes) -> int:
    return int.from_bytes(raw, "little", signed=True)


def crc(raw: bytes) -> int:
    return int(koopman_crc32(np.frombuffer(raw, dtype=np.uint8), np.array([0]), np.array([len(raw)]))[0])


# ---------------------------------------------------------------------------------------------------------------
# Writing MEF 2 files from real blocks
# ---------------------------------------------------------------------------------------------------------------


def read_mef2_file(path: Path, numbers: list[int] | None = None) -> tuple[bytes, list[bytes]]:
    """A MEF 2 file's header and its blocks, or those numbered, each with the padding before the next one."""
    with open(path, "rb") as file:
        header = file.read(HEADER_BYTES)
        index_offset, number_of_blocks = struct.unpack_from("<QQ", header, 816)
        file.seek(index_offset)
        offsets = list(np.frombuffer(file.read(24 * number_of_blocks), dtype="<u8")[1::3].astype(int))
        ends = offsets[1:] + [index_offset]
        blocks = []
        for number in range(number_of_blocks) if numbers is None else numbers:
            file.seek(offsets[number])
            blocks.append(file.read(ends[number] - offsets[number]))
    return header, blocks


def patch_block(block: bytes, *, start_time: int | None = None, flag: bool | None = None) -> bytes:
    """A block with a new start time or discontinuity flag, and its CRC redone."""
    block = bytearray(block)
    if start_time is not None:
        struct.pack_into("<Q", block, 8, start_time)
    if flag is not None:
        block[30] = (block[30] | 1) if flag else (block[30] & ~1)
    compressed_bytes = struct.unpack_from("<I", block, 4)[0]
    struct.pack_into("<I", block, 0, crc(bytes(block[4 : BLOCK_HEADER_BYTES + compressed_bytes])))
    return bytes(block)


# Text fields of the header, by offset and length (mef.h).
TEXT_FIELDS = {
    "institution": (0, 64),
    "unencrypted_text_field": (64, 64),
    "subject_first_name": (176, 32),
    "subject_second_name": (208, 32),
    "subject_third_name": (240, 32),
    "subject_id": (272, 32),
    "session_password": (304, 16),
    "channel_name": (376, 32),
    "acquisition_system": (464, 32),
    "channel_comments": (496, 128),
    "study_comments": (624, 128),
    "compression_algorithm": (756, 32),
    "anonymized_subject_name": (956, 64),
}
NUMBER_FIELDS = {
    "subject_encryption_used": (160, "B"),
    "session_encryption_used": (161, "B"),
    "data_encryption_used": (162, "B"),
    "byte_order_code": (163, "B"),
    "header_version_major": (164, "B"),
    "header_version_minor": (165, "B"),
    "sampling_frequency": (424, "d"),
    "voltage_conversion_factor": (456, "d"),
    "physical_channel_number": (752, "i"),
    "gmt_offset": (836, "f"),
}


def patch_header(header: bytes, **fields) -> bytes:
    """A header with fields replaced; ``write_mef2`` redoes the CRC."""
    header = bytearray(header)
    for name, value in fields.items():
        if name in TEXT_FIELDS:
            offset, length = TEXT_FIELDS[name]
            encoded = value.encode("latin-1")[: length - 1]
            header[offset : offset + length] = encoded + b"\0" * (length - len(encoded))
        else:
            offset, code = NUMBER_FIELDS[name]
            struct.pack_into("<" + code, header, offset, value)
    return bytes(header)


def write_mef2(path: Path, header: bytes, blocks: list[bytes], *, discontinuity_index: bool = True, header_crc=True):
    """
    Write a MEF 2 file from a header and blocks, as mef_lib's ``write_mef_ind`` lays one out.

    The block index is built from the block headers. A MEF 2.1 file also gets the discontinuity index, of the
    1-based numbers of its flagged blocks, unless ``discontinuity_index`` is False, and its header CRC.
    """
    header = bytearray(header)
    data = b"".join(blocks)
    entries, offset, sample = [], HEADER_BYTES, 0
    for block in blocks:
        start_time = struct.unpack_from("<Q", block, 8)[0]
        entries.append(struct.pack("<QQQ", start_time, offset, sample))
        offset += len(block)
        sample += struct.unpack_from("<I", block, 20)[0]
    index_offset = HEADER_BYTES + len(data)
    flagged = [number + 1 for number, block in enumerate(blocks) if block[30] & 1]
    minor = header[165]
    write_discontinuities = minor >= 1 and discontinuity_index and flagged
    discontinuities = struct.pack(f"<{len(flagged)}Q", *flagged) if write_discontinuities else b""
    first_time = struct.unpack_from("<Q", blocks[0], 8)[0]
    struct.pack_into("<Q", header, 368, sample)
    struct.pack_into("<QQ", header, 408, first_time, struct.unpack_from("<Q", blocks[-1], 8)[0])
    struct.pack_into("<QQ", header, 816, index_offset, len(blocks))
    if minor >= 1:
        disc_offset = index_offset + 24 * len(blocks) if write_discontinuities else 0
        struct.pack_into("<QQ", header, 840, disc_offset, len(flagged) if write_discontinuities else 0)
        if header_crc:
            struct.pack_into("<I", header, 1020, crc(bytes(header[:1020])))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(header) + data + b"".join(entries) + discontinuities)


def copy_recording(source: Path, destination: Path) -> Path:
    shutil.copytree(source, destination)
    return destination


def rewrite(path: Path, *, header=None, blocks=None, **kwargs):
    """Rewrite a fixture file in place with a changed header or blocks."""
    original_header, original_blocks = read_mef2_file(path)
    write_mef2(path, header or original_header, blocks or original_blocks, **kwargs)


def block_statistics(path: Path):
    """Each block's stored maximum, minimum and CRC, and the CRC its bytes have."""
    _, blocks = read_mef2_file(path)
    rows = []
    for block in blocks:
        compressed_bytes = struct.unpack_from("<I", block, 4)[0]
        rows.append(
            (
                si3(block[24:27]),
                si3(block[27:30]),
                struct.unpack_from("<I", block, 0)[0],
                crc(block[4 : BLOCK_HEADER_BYTES + compressed_bytes]),
                struct.unpack_from("<I", block, 20)[0],
            )
        )
    return rows


def quiet(function, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return function(*args, **kwargs)


# ---------------------------------------------------------------------------------------------------------------
# The real excerpts
# ---------------------------------------------------------------------------------------------------------------


class TestRealExcerpts:
    @pytest.mark.parametrize("folder", [PENN, AD3, AD3_STEP])
    def test_blocks_hold_what_their_headers_state(self, folder):
        """The excerpts' blocks are the source files' bytes: their CRCs hold and their stored max and min are real."""
        (path,) = folder.glob("*.mef")
        for maximum, minimum, stored_crc, computed_crc, samples in block_statistics(path):
            assert stored_crc == computed_crc
            assert samples > 0 and maximum >= minimum

    @pytest.mark.parametrize("use_numba", DECODERS)
    @pytest.mark.parametrize("folder", [PENN, AD3, AD3_STEP])
    def test_decoded_blocks_match_stored_maximum_and_minimum(self, folder, use_numba):
        (path,) = folder.glob("*.mef")
        header, blocks = read_mef2_file(path)
        buffer = np.frombuffer(b"".join(blocks), dtype=np.uint8)
        starts = np.cumsum([0] + [len(block) for block in blocks[:-1]])
        statistics = block_statistics(path)
        counts = np.array([row[4] for row in statistics])
        samples = decode_red_blocks(
            buffer,
            block_starts=starts,
            block_bytes=np.array([BLOCK_HEADER_BYTES + struct.unpack_from("<I", block, 4)[0] for block in blocks]),
            difference_counts=np.array([struct.unpack_from("<I", block, 16)[0] + 1 for block in blocks]),
            sample_counts=counts,
            scale_factors=np.ones(len(blocks)),
            detrend_slopes=np.zeros(len(blocks)),
            detrend_intercepts=np.zeros(len(blocks)),
            layout=MEF2_RED_LAYOUT,
            use_numba=use_numba,
        )
        edges = np.concatenate([[0], np.cumsum(counts)[:-1]])
        np.testing.assert_array_equal(np.maximum.reduceat(samples, edges), [row[0] for row in statistics])
        np.testing.assert_array_equal(np.minimum.reduceat(samples, edges), [row[1] for row in statistics])

    @pytest.mark.parametrize("folder", [PENN, AD3, AD3_STEP])
    def test_samples_match_the_full_source_file(self, folder):
        recording = quiet(read_mef2_recording, folder, validate_crc=True)
        (channel,) = recording.channels
        expected = EXPECTED[folder.name]
        samples = channel.read_blocks(0, len(channel.blocks))
        assert samples.size == expected["number_of_samples"]
        assert digest(samples) == expected["sha256"]
        assert [run.number_of_samples for run in recording.runs] == expected["run_samples"]
        assert [run.start_time for run in recording.runs] == expected["run_start_times"]

    def test_mef_2_1_header(self):
        header = read_mef2_header(PENN / "LI07.mef")
        assert header["version"] == "2.1"
        assert header["channel_name"] == "LI07"
        assert header["sampling_frequency"] == 1024.0
        # PennEPI stores the factor negated: the counts are inverted.
        assert header["voltage_conversion_factor"] == pytest.approx(-0.004153729475433701)
        assert header["gmt_offset"] == -6.0
        assert header["number_of_discontinuity_entries"] == 9

    def test_mef_2_0_header(self):
        header = read_mef2_header(AD3 / "AD3.mef")
        assert header["version"] == "2.0"
        assert header["sampling_frequency"] == 500.0
        assert header["voltage_conversion_factor"] == pytest.approx(0.2658386864277569)
        assert header["acquisition_system"] == "XLTek EEG 128"
        # MEF 2.0 has neither a GMT offset nor a discontinuity index.
        assert header["gmt_offset"] is None and header["number_of_discontinuity_entries"] == 0

    def test_placeholder_block_is_dropped(self):
        recording = read_mef2_recording(PENN)
        (channel,) = recording.channels
        header, blocks = read_mef2_file(PENN / "LI07.mef")
        assert channel.dropped_placeholder_block
        assert len(channel.blocks) == len(blocks) - 1
        assert struct.unpack_from("<I", blocks[0], 20)[0] == 1
        # The recording starts at the first real block, 2.5 s after the placeholder.
        assert recording.start_time == struct.unpack_from("<Q", blocks[1], 8)[0]
        assert recording.start_time - struct.unpack_from("<Q", header, 408)[0] == 2_544_921

    def test_dropouts_split_runs(self):
        """PennEPI's flagged blocks follow real dropouts of whole samples, each starting its own run."""
        recording = read_mef2_recording(PENN)
        assert [run.number_of_blocks for run in recording.runs] == [2, 1, 1, 1, 1, 1, 1, 2]
        blocks = recording.channels[0].blocks
        gaps = [
            (blocks["start_time"][run.first_block] - blocks["start_time"][run.first_block - 1]) / 1e6
            - blocks["number_of_samples"][run.first_block - 1] / 1024
            for run in recording.runs[2:]
        ]
        np.testing.assert_allclose(gaps, [0.01172, 0.48242, 0.04395, 0.55762, 0.05762, 230.4834], atol=1e-4)

    def test_discontinuity_index_agrees_with_block_flags(self, tmp_path):
        """The 2.1 index counts blocks from 1; without it the block headers' flags give the same runs."""
        with_index = read_mef2_recording(PENN)
        copy = copy_recording(PENN, tmp_path / "no_index")
        rewrite(copy / "LI07.mef", discontinuity_index=False)
        assert read_mef2_header(copy / "LI07.mef")["number_of_discontinuity_entries"] == 0
        without_index = read_mef2_recording(copy)
        assert without_index.runs == with_index.runs
        np.testing.assert_array_equal(
            without_index.channels[0].blocks["discontinuity"], with_index.channels[0].blocks["discontinuity"]
        )

    def test_deidentified_date_warns(self):
        interface = MEF2RecordingInterface(AD3)
        with pytest.warns(
            UserWarning, match="starts at 2000-01-01T00:00:00\\+00:00, on the date de-identified"
        ) as record:
            interface.get_metadata()
        # Attributed to this file, not to neuroconv.
        assert record[0].filename == __file__

    def test_mef_2_0_runs_follow_block_flags(self):
        recording = read_mef2_recording(AD3)
        assert [run.number_of_blocks for run in recording.runs] == [5, 3]
        assert recording.runs[1].start_time - recording.runs[0].start_time == 1_224_664_000

    def test_clock_step_back_stays_in_the_run(self):
        with pytest.warns(UserWarning, match=r"go back by more than a second at 1 blocks .*block 2 \(-1\.017 s\)"):
            recording = read_mef2_recording(AD3_STEP)
        assert len(recording.runs) == 1


# ---------------------------------------------------------------------------------------------------------------
# Runs, from patched copies
# ---------------------------------------------------------------------------------------------------------------


class TestRuns:
    def shifted(self, tmp_path, block_number, *, seconds=0.0, flag=None, folder=AD3):
        """A copy of a recording with the times of a block and those after it moved, and the block's flag set or cleared."""
        copy = copy_recording(folder, tmp_path / "copy")
        (path,) = copy.glob("*.mef")
        header, blocks = read_mef2_file(path)
        for number in range(block_number, len(blocks)):
            start_time = struct.unpack_from("<Q", blocks[number], 8)[0] + round(seconds * 1e6)
            blocks[number] = patch_block(
                blocks[number], start_time=start_time, flag=flag if number == block_number else None
            )
        write_mef2(path, header, blocks)
        return copy

    def test_flagged_block_that_continues_the_run(self, tmp_path):
        """Writers flag blocks that follow nothing missing; one starting where the last ended continues the run."""
        copy = self.shifted(tmp_path, 2, flag=True)
        assert [run.number_of_blocks for run in read_mef2_recording(copy).runs] == [5, 3]

    def test_flagged_gap_starts_a_run(self, tmp_path):
        copy = self.shifted(tmp_path, 2, seconds=0.1, flag=True)
        recording = read_mef2_recording(copy)
        assert [run.number_of_blocks for run in recording.runs] == [2, 3, 3]
        assert recording.runs[1].start_time == recording.channels[0].blocks["start_time"][2]

    def test_unflagged_jump_forward_starts_a_run(self, tmp_path):
        copy = self.shifted(tmp_path, 2, seconds=5.0)
        with pytest.warns(
            UserWarning, match=r"jump forward by more than a second at 1 blocks .*: block 2 \(\+5\.000 s\)"
        ):
            recording = read_mef2_recording(copy)
        assert [run.number_of_blocks for run in recording.runs] == [2, 3, 3]

    def test_unflagged_jitter_stays_in_the_run(self, tmp_path):
        copy = self.shifted(tmp_path, 2, seconds=0.06)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert [run.number_of_blocks for run in read_mef2_recording(copy).runs] == [5, 3]

    def test_flagged_block_back_in_time_overlaps(self, tmp_path):
        copy = self.shifted(tmp_path, 6, seconds=-2.0, flag=True)
        with pytest.warns(UserWarning, match=r"1 runs .* start before the run before them ends \(runs \[2\]\)"):
            recording = read_mef2_recording(copy)
        assert [run.number_of_blocks for run in recording.runs] == [5, 1, 2]


# ---------------------------------------------------------------------------------------------------------------
# Headers, refusals, CRCs
# ---------------------------------------------------------------------------------------------------------------


class TestHeaders:
    def patched(self, tmp_path, folder=PENN, header_crc=True, **fields):
        copy = copy_recording(folder, tmp_path / "copy")
        (path,) = copy.glob("*.mef")
        header, blocks = read_mef2_file(path)
        write_mef2(path, patch_header(header, **fields), blocks, header_crc=header_crc)
        return copy

    @pytest.mark.parametrize(
        "fields, message",
        [
            (dict(session_encryption_used=1), "is encrypted"),
            (dict(data_encryption_used=1), "is encrypted"),
            (dict(header_version_minor=2), "MEF 2.2 file. Only MEF 2.0 and 2.1"),
            (dict(header_version_major=3), "does not have a MEF 2 header"),
            (dict(byte_order_code=0), "big-endian"),
            (dict(compression_algorithm="None"), "not RED"),
        ],
    )
    def test_refused(self, tmp_path, fields, message):
        copy = self.patched(tmp_path, **fields)
        with pytest.raises(ValueError, match=message):
            read_mef2_recording(copy)

    def test_subject_encryption_alone_is_read(self, tmp_path):
        """It covers only the subject fields, which are never read."""
        copy = self.patched(tmp_path, subject_encryption_used=1)
        assert len(read_mef2_recording(copy).runs) == 8

    def test_not_a_mef_2_recording(self, tmp_path):
        with pytest.raises(ValueError, match="holds no .mef files"):
            read_mef2_recording(tmp_path)
        (tmp_path / "short.mef").write_bytes(b"\0" * 100)
        with pytest.raises(ValueError, match="shorter than a MEF 2 header"):
            read_mef2_recording(tmp_path)

    def test_header_crc(self, tmp_path):
        copy = self.patched(tmp_path, header_crc=False, acquisition_system="changed after the CRC")
        with pytest.warns(UserWarning, match="fails its CRC"):
            read_mef2_recording(copy)
        with pytest.raises(ValueError, match="fails its CRC"):
            read_mef2_recording(copy, validate_crc=True)

    def test_block_crc(self, tmp_path):
        copy = copy_recording(AD3, tmp_path / "copy")
        path = copy / "AD3.mef"
        header, blocks = read_mef2_file(path)
        corrupted = bytearray(blocks[3])
        corrupted[BLOCK_HEADER_BYTES + 50] ^= 0xFF
        blocks[3] = bytes(corrupted)
        write_mef2(path, header, blocks)
        recording = read_mef2_recording(copy, validate_crc=True)
        with pytest.raises(ValueError, match="Block 3 of the MEF 2 file .* fails its CRC"):
            recording.channels[0].read_blocks(0, 5)
        # Blocks around it still read.
        assert recording.channels[0].read_blocks(4, 1).size == recording.channels[0].blocks["number_of_samples"][4]

    def test_index_that_disagrees_with_the_blocks(self, tmp_path):
        copy = copy_recording(AD3, tmp_path / "copy")
        path = copy / "AD3.mef"
        raw = bytearray(path.read_bytes())
        index_offset = struct.unpack_from("<Q", raw, 816)[0]
        # Move block 2's start time in the index alone.
        struct.pack_into("<Q", raw, index_offset + 2 * 24, struct.unpack_from("<Q", raw, index_offset + 2 * 24)[0] + 7)
        path.write_bytes(bytes(raw))
        recording = quiet(read_mef2_recording, copy)
        with pytest.raises(ValueError, match="Block 2 of the MEF 2 file .* does not have the start time"):
            recording.channels[0].read_blocks(0, 5)

    def test_index_out_of_order(self, tmp_path):
        copy = copy_recording(AD3, tmp_path / "copy")
        path = copy / "AD3.mef"
        raw = bytearray(path.read_bytes())
        index_offset = struct.unpack_from("<Q", raw, 816)[0]
        struct.pack_into("<Q", raw, index_offset + 3 * 24 + 8, 1024)
        path.write_bytes(bytes(raw))
        with pytest.raises(ValueError, match="not in file order"):
            read_mef2_recording(copy)


# ---------------------------------------------------------------------------------------------------------------
# Several channels
# ---------------------------------------------------------------------------------------------------------------


def channel_copy(source: Path, destination: Path, **header_fields):
    """Write ``source``'s single file to ``destination`` with header fields replaced."""
    header, blocks = read_mef2_file(next(source.glob("*.mef")))
    write_mef2(destination, patch_header(header, **header_fields), blocks)


class TestChannels:
    def test_order_names_and_skipping(self, tmp_path):
        folder = tmp_path / "recording"
        channel_copy(AD3, folder / "b.mef", channel_name="LTG2", physical_channel_number=7)
        channel_copy(AD3, folder / "a.mef", channel_name="LTG10", physical_channel_number=12)
        channel_copy(AD3, folder / "c.mef", channel_name="LTG1", physical_channel_number=3)
        assert MEF2RecordingInterface.get_available_channel_ids(folder) == ["LTG1", "LTG2", "LTG10"]
        recording = read_mef2_recording(folder, channels_to_skip=["LTG2"])
        assert recording.channel_ids == ["LTG1", "LTG10"]
        with pytest.raises(ValueError, match=r"not in the recording: \['LTG3'\]"):
            read_mef2_recording(folder, channels_to_skip=["LTG3"])

    def test_a_file_that_cannot_be_read_can_be_skipped(self, tmp_path):
        folder = tmp_path / "recording"
        channel_copy(AD3, folder / "AD3.mef")
        channel_copy(AD3, folder / "secret.mef", session_encryption_used=1, channel_name="garbled")
        with pytest.raises(ValueError, match="is encrypted"):
            read_mef2_recording(folder)
        # Named by its file name, its session fields being encrypted.
        assert MEF2RecordingInterface.get_available_channel_ids(folder) == ["AD3", "secret"]
        assert read_mef2_recording(folder, channels_to_skip=["secret"]).channel_ids == ["AD3"]

    def test_empty_channel_name_falls_back_to_the_file_name(self, tmp_path):
        channel_copy(AD3, tmp_path / "recording" / "RAH4.mef", channel_name="")
        assert read_mef2_recording(tmp_path / "recording").channel_ids == ["RAH4"]

    def test_repeated_names_are_renamed(self, tmp_path):
        folder = tmp_path / "recording"
        channel_copy(AD3, folder / "a.mef", channel_name="EKG", physical_channel_number=1)
        channel_copy(AD3, folder / "b.mef", channel_name="EKG", physical_channel_number=2)
        with pytest.warns(UserWarning, match=r"'EKG' -> 'EKG_2'"):
            extractor = MEF2RecordingExtractor(folder)
        assert list(extractor.get_channel_ids()) == ["EKG", "EKG_2"]
        assert list(extractor.get_property("original_channel_name")) == ["EKG", "EKG"]
        assert list(MEF2RecordingExtractor(folder, channels_to_skip=["EKG_2"]).get_channel_ids()) == ["EKG"]

    def test_streams(self, tmp_path):
        folder = tmp_path / "recording"
        channel_copy(PENN, folder / "LI07.mef")
        channel_copy(PENN, folder / "ECG.mef", channel_name="ECG", sampling_frequency=512.0)
        assert MEF2RecordingInterface.get_stream_names(folder) == ["512 Hz", "1024 Hz"]
        with pytest.raises(
            ValueError, match=r"2 different rates .*choose one with stream_name, one of \['1024 Hz', '512 Hz'\]"
        ):
            read_mef2_recording(folder)
        assert read_mef2_recording(folder, stream_name="1024 Hz").channel_ids == ["LI07"]
        # The 1024 Hz file relabelled 512 Hz: its block times no longer match its sample counts, which warns.
        assert quiet(read_mef2_recording, folder, stream_name="512 Hz").channel_ids == ["ECG"]
        assert read_mef2_recording(folder, channels_to_skip=["ECG"]).channel_ids == ["LI07"]
        with pytest.raises(ValueError, match="stream_name '256 Hz' is not a stream"):
            read_mef2_recording(folder, stream_name="256 Hz")

    def test_channels_must_share_runs(self, tmp_path):
        folder = tmp_path / "recording"
        header, blocks = read_mef2_file(AD3 / "AD3.mef")
        write_mef2(folder / "AD3.mef", header, blocks)
        write_mef2(
            folder / "AD4.mef", patch_header(header, channel_name="AD4", physical_channel_number=84), blocks[:-1]
        )
        with pytest.raises(ValueError, match="were not recorded over the same stretches of time"):
            read_mef2_recording(folder)
        assert read_mef2_recording(folder, channels_to_skip=["AD4"]).channel_ids == ["AD3"]


# ---------------------------------------------------------------------------------------------------------------
# Extractor and interface
# ---------------------------------------------------------------------------------------------------------------


class TestExtractor:
    def test_traces_across_blocks_and_runs(self, tmp_path):
        folder = tmp_path / "recording"
        channel_copy(PENN, folder / "LI07.mef")
        channel_copy(PENN, folder / "LI08.mef", channel_name="LI08", physical_channel_number=99)
        recording = MEF2RecordingExtractor(folder)
        full = recording.mef2_recording.channels[0].read_blocks(0, len(recording.mef2_recording.channels[0].blocks))
        assert recording.get_num_segments() == 8
        assert recording.get_dtype() == np.int32
        edges = np.concatenate([[0], np.cumsum([run.number_of_samples for run in recording.mef2_recording.runs])])
        for segment in range(8):
            traces = recording.get_traces(segment_index=segment)
            np.testing.assert_array_equal(traces[:, 0], full[edges[segment] : edges[segment + 1]])
            np.testing.assert_array_equal(traces[:, 1], traces[:, 0])
        # A window straddling the block boundary at sample 2500 of the first run.
        np.testing.assert_array_equal(
            recording.get_traces(segment_index=0, start_frame=2490, end_frame=2510, channel_ids=["LI08"])[:, 0],
            full[2490:2510],
        )
        assert recording.get_start_time(segment_index=7) == pytest.approx(
            (recording.mef2_recording.runs[7].start_time - recording.mef2_recording.start_time) / 1e6
        )

    def test_gains_keep_their_sign(self):
        recording = MEF2RecordingExtractor(PENN)
        np.testing.assert_allclose(recording.get_channel_gains(), [-0.004153729475433701])
        assert list(recording.get_property("physical_unit")) == ["uV"]
        raw = recording.get_traces(segment_index=0, start_frame=0, end_frame=10)
        scaled = recording.get_traces(segment_index=0, start_frame=0, end_frame=10, return_in_uV=True)
        np.testing.assert_allclose(scaled, raw * -0.004153729475433701)

    def test_missing_conversion_factor(self, tmp_path):
        channel_copy(AD3, tmp_path / "recording" / "AD3.mef", voltage_conversion_factor=0.0)
        with pytest.warns(UserWarning, match="give no voltage conversion factor"):
            recording = MEF2RecordingExtractor(tmp_path / "recording")
        np.testing.assert_array_equal(recording.get_channel_gains(), [1.0])


def annotations_xml(*annotations) -> str:
    """An IEEG.org annotations file, each annotation given as (type, start offset µs, end offset µs, [files])."""
    elements = []
    for number, (annotation_type, start, end, files) in enumerate(annotations):
        channels = "".join(f'<channel fileName="{name}" id="x{index}"/>' for index, name in enumerate(files))
        elements.append(
            f'<annotation id="{number}" type="{annotation_type}" startOffsetUsecs="{start}" endOffsetUsecs="{end}" '
            f'layer="Test layer" annotator="manual" creatorId="1"><channels>{channels}</channels></annotation>'
        )
    return '<?xml version="1.0" encoding="UTF-8"?><annotations>' + "".join(elements) + "</annotations>"


class TestInterface:
    def test_conversion(self, tmp_path):
        with pytest.warns(UserWarning, match="go back by more than a second"):
            interface = MEF2RecordingInterface(AD3_STEP)
        # Cut from 27 hours into a recording moved to 2000-01-01, so not itself on that date.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            metadata = interface.get_metadata()
        assert metadata["NWBFile"]["session_start_time"] == datetime(
            2000, 1, 2, 10, 40, 21, 342000, tzinfo=timezone.utc
        )
        nwbfile_path = tmp_path / "step.nwb"
        interface.run_conversion(nwbfile_path=nwbfile_path, metadata=metadata, overwrite=True)
        full = interface.mef2_recording.channels[0].read_blocks(0, 5)
        with NWBHDF5IO(nwbfile_path, "r") as io:
            nwbfile = io.read()
            (series,) = nwbfile.acquisition.values()
            assert series.rate == 500.0 and series.starting_time == 0.0
            assert series.data.dtype == np.int32
            np.testing.assert_array_equal(series.data[:, 0], full)
            assert series.conversion == pytest.approx(0.2658386864277569e-6)
            assert list(nwbfile.electrodes["channel_name"][:]) == ["AD3"]
            # One run: no runs table.
            assert "runs" not in (nwbfile.intervals or {})

    def test_negative_gain_round_trips(self, tmp_path):
        interface = MEF2RecordingInterface(PENN)
        nwbfile_path = tmp_path / "penn.nwb"
        interface.run_conversion(nwbfile_path=nwbfile_path, metadata=quiet(interface.get_metadata), overwrite=True)
        with NWBHDF5IO(nwbfile_path, "r") as io:
            nwbfile = io.read()
            series = sorted(nwbfile.acquisition.values(), key=lambda series: series.starting_time)
            assert len(series) == 8
            assert series[0].conversion == pytest.approx(-0.004153729475433701e-6)
            counts = series[0].data[:10, 0]
            np.testing.assert_allclose(series[0].get_data_in_units()[:10, 0], counts * -0.004153729475433701e-6)
            runs = nwbfile.intervals["runs"].to_dataframe()
            np.testing.assert_allclose(runs["start_time"], [item.starting_time for item in series])
            assert list(runs["source_file"].unique()) == [PENN.name]
            # The time zone the header states.
            assert nwbfile.session_start_time.utcoffset() == timedelta(hours=-6)

    def test_annotations_are_placed_on_their_samples(self, tmp_path):
        """IEEG.org times are offsets from the header start; they are placed through the block index."""
        folder = copy_recording(AD3_STEP, tmp_path / "step")
        recording = quiet(read_mef2_recording, folder)
        blocks = recording.channels[0].blocks
        counts = blocks["number_of_samples"]
        start = int(blocks["start_time"][0])  # also the header's recording start time
        # 1 s into block 0; 0.5 s into block 2, a time block 1 also claims, the clock having gone back.
        in_block_0 = 1_000_000
        in_block_2 = int(blocks["start_time"][2]) - start + 500_000
        assert blocks["start_time"][1] + counts[1] * 2000 > start + in_block_2
        (folder / "annotations.xml").write_text(
            annotations_xml(
                ("Seizure", in_block_2, in_block_2 + 2_000_000, ["AD3.mef", "AD4.mef"]),
                ("Note", in_block_0, in_block_0, []),
            ),
            encoding="utf-8",
        )
        interface = quiet(MEF2RecordingInterface, folder)
        nwbfile = mock_NWBFile()
        interface.add_to_nwbfile(nwbfile=nwbfile, metadata=quiet(interface.get_metadata))
        table = nwbfile.intervals["annotations"].to_dataframe()
        assert list(table["label"]) == ["Note", "Seizure"]
        # The note spans one sample; the seizure starts after the samples of blocks 0 and 1.
        np.testing.assert_allclose(table["start_time"], [1.0, (counts[0] + counts[1]) / 500 + 0.5])
        np.testing.assert_allclose(table["stop_time"], [1.002, (counts[0] + counts[1]) / 500 + 2.5])
        assert list(table["channels"]) == ["", "AD3, AD4"]
        assert list(table["layer"]) == ["Test layer"] * 2
        assert set(table["source_file"]) == {"annotations.xml"}

        nwbfile = mock_NWBFile()
        interface.add_to_nwbfile(nwbfile=nwbfile, metadata=quiet(interface.get_metadata), write_annotations=False)
        assert nwbfile.intervals is None or "annotations" not in nwbfile.intervals

    def test_annotations_follow_alignment_and_gaps(self, tmp_path):
        folder = copy_recording(AD3, tmp_path / "ad3")
        recording = read_mef2_recording(folder)
        second_run = recording.runs[1].start_time - recording.recording_start_time
        annotations_path = tmp_path / "elsewhere.xml"
        # 10 s into the gap between the runs, and 1 s into the second run.
        gap_time = recording.runs[0].number_of_samples * 2000 + 10_000_000
        annotations_path.write_text(
            annotations_xml(
                ("Gap", gap_time, gap_time + 1, []), ("Event", second_run + 1_000_000, second_run + 1_000_001, [])
            ),
            encoding="utf-8",
        )
        interface = MEF2RecordingInterface(folder, annotations_file_path=annotations_path)
        interface.set_aligned_segment_starting_times([100.0, 2000.0])
        nwbfile = mock_NWBFile()
        interface.add_to_nwbfile(nwbfile=nwbfile, metadata=quiet(interface.get_metadata))
        starts = list(nwbfile.intervals["annotations"]["start_time"][:])
        # The base class shifts each run by the value given: the second run moves from 1224.664 s to 3224.664 s.
        # A time in a gap is placed from the start of the run before it.
        assert starts == pytest.approx([100.0 + gap_time / 1e6, 3225.664])
        np.testing.assert_allclose(nwbfile.intervals["runs"]["start_time"][:], [100.0, 3224.664])

    def test_annotations_file(self):
        recording = read_mef2_recording(AD3)
        (annotation,) = read_ieeg_annotations(
            FIXTURES / "Discover13_annotations_excerpt.xml", recording.recording_start_time
        )
        assert annotation.annotation_type == "Seizure"
        assert annotation.end_time - annotation.start_time == 50_606_000
        assert annotation.channels[:3] == ["AD1", "AD2", "AD3"]

    def test_phi_never_reaches_the_nwb_file(self, tmp_path):
        """The subject, institution and comment fields are never read."""
        sentinels = dict(
            institution="PHI-INSTITUTION",
            unencrypted_text_field="PHI-UNENCRYPTED-TEXT",
            subject_first_name="PHI-FIRST",
            subject_second_name="PHI-SECOND",
            subject_third_name="PHI-THIRD",
            subject_id="PHI-SUBJECT-ID",
            session_password="PHI-PASSWORD",
            channel_comments="PHI-CHANNEL-COMMENTS",
            study_comments="PHI-STUDY-COMMENTS",
            anonymized_subject_name="PHI-ANONYMIZED",
        )
        folder = tmp_path / "recording"
        channel_copy(PENN, folder / "LI07.mef", **sentinels)
        interface = MEF2RecordingInterface(folder)
        metadata = quiet(interface.get_metadata)
        assert "Subject" not in metadata and "PHI-" not in repr(metadata)
        nwbfile_path = tmp_path / "phi.nwb"
        interface.run_conversion(nwbfile_path=nwbfile_path, metadata=metadata, overwrite=True)
        written = nwbfile_path.read_bytes()
        assert b"PHI-" not in written
        assert b"PHI-FIRST" in (folder / "LI07.mef").read_bytes()
        with NWBHDF5IO(nwbfile_path, "r") as io:
            assert io.read().subject is None

    def test_signature_is_keyword_only_after_folder_path(self):
        import inspect

        parameters = inspect.signature(MEF2RecordingInterface.__init__).parameters
        assert [
            name for name, parameter in parameters.items() if parameter.kind == parameter.POSITIONAL_OR_KEYWORD
        ] == [
            "self",
            "folder_path",
        ]


class TestKoopmanCRC:
    @pytest.mark.parametrize("use_numba", DECODERS)
    def test_matches_stored_crcs(self, use_numba):
        header, blocks = read_mef2_file(PENN / "LI07.mef")
        buffer = np.frombuffer(b"".join(blocks), dtype=np.uint8)
        starts = np.cumsum([0] + [len(block) for block in blocks[:-1]])
        ends = starts + np.array([BLOCK_HEADER_BYTES + struct.unpack_from("<I", block, 4)[0] for block in blocks])
        crcs = koopman_crc32(buffer, starts + 4, ends, use_numba=use_numba)
        assert list(crcs) == [struct.unpack_from("<I", block, 0)[0] for block in blocks]
        assert crc(header[:1020]) == struct.unpack_from("<I", header, 1020)[0]


class TestMEF2RecordingInterfaceContract(RecordingExtractorInterfaceTestMixin):
    data_interface_cls = MEF2RecordingInterface
    interface_kwargs = dict(folder_path=str(PENN))

    def check_extracted_metadata(self, metadata: dict):
        assert metadata["NWBFile"]["session_start_time"].date() == datetime(2000, 1, 1).date()
        assert "mef2_recording" in metadata["Ecephys"]["ElectricalSeries"]


class TestMEF2_0RecordingInterfaceContract(RecordingExtractorInterfaceTestMixin):
    data_interface_cls = MEF2RecordingInterface
    interface_kwargs = dict(folder_path=str(AD3))

    def check_extracted_metadata(self, metadata: dict):
        assert metadata["NWBFile"]["session_start_time"] == datetime(2000, 1, 1, tzinfo=timezone.utc)
