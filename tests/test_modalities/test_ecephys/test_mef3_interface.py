"""Tests for the MEF3RecordingInterface, its reader and the RED decoder.

The sessions in ``mef3_fixtures/`` were written, or cut from ds004696, by ``build_mef3_fixtures.py``, which
also stored in ``expected.json`` what pymef reads from them. pymef is not needed to run these tests.
"""

import hashlib
import json
import re
import shutil
import struct
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from pynwb import NWBHDF5IO
from pynwb.testing.mock.file import mock_NWBFile

from neuroconv.datainterfaces import MEF3RecordingInterface
from neuroconv.datainterfaces.ecephys.mef._mef3_extractor import (
    MEF3RecordingExtractor,
)
from neuroconv.datainterfaces.ecephys.mef._mef3_reader import (
    UUTC_NO_ENTRY,
    _record_body_fields,
    _true_time,
    read_mef3_session,
)
from neuroconv.datainterfaces.ecephys.mef._red import (
    MEF3_RED_LAYOUT,
    NUMBA_AVAILABLE,
    REDDecodeError,
    decode_red_blocks,
)
from neuroconv.tools.testing.data_interface_mixins import (
    RecordingExtractorInterfaceTestMixin,
)

FIXTURES = Path(__file__).parent / "mef3_fixtures"
EXPECTED = json.loads((FIXTURES / "expected.json").read_text())
GAPPED = FIXTURES / "gapped.mefd"
LOSSY = FIXTURES / "lossy.mefd"
ENCRYPTED = FIXTURES / "encrypted.mefd"
TRUNCATED = FIXTURES / "ds004696_sub-06_truncated.mefd"
CONTIGUOUS = FIXTURES / "contiguous.mefd"
START_TIME = 946684800000000  # the µUTC start of the synthetic sessions, 2000-01-01T00:00:00 UTC

DECODERS = [
    pytest.param(True, id="numba", marks=pytest.mark.skipif(not NUMBA_AVAILABLE, reason="numba is not installed")),
    pytest.param(False, id="numpy"),
]


def digest(samples) -> str:
    return hashlib.sha256(np.asarray(samples, dtype="<i4").tobytes()).hexdigest()


def read_all_blocks(channel, use_numba=None) -> np.ndarray:
    """Every sample of a channel, decoded in one call per data file, bypassing the extractor."""
    if use_numba is None:
        return channel.read_blocks(0, len(channel.blocks))
    decoded = []
    for segment, data_path in enumerate(channel.data_paths):
        blocks = channel.blocks[channel.blocks["segment"] == segment]
        decoded.append(
            decode_red_blocks(
                np.fromfile(data_path, dtype=np.uint8),
                block_starts=blocks["file_offset"],
                block_bytes=blocks["block_bytes"],
                difference_counts=blocks["difference_bytes"],
                sample_counts=blocks["number_of_samples"],
                scale_factors=blocks["scale_factor"],
                detrend_slopes=blocks["detrend_slope"],
                detrend_intercepts=blocks["detrend_intercept"],
                layout=MEF3_RED_LAYOUT,
                use_numba=use_numba,
            )
        )
    return np.concatenate(decoded)


def set_discontinuity_flag(session_path: Path, segment: int, block: int, flagged: bool):
    """Set or clear one block's discontinuity flag, in its index entry and in its RED header, in every channel."""
    for channel_directory in session_path.glob("*.timd"):
        stem = f"{channel_directory.name[:-len('.timd')]}-{segment:06d}"
        segment_directory = channel_directory / f"{stem}.segd"
        index_path, data_path = segment_directory / f"{stem}.tidx", segment_directory / f"{stem}.tdat"
        index, data = bytearray(index_path.read_bytes()), bytearray(data_path.read_bytes())
        entry = 1024 + 56 * block
        offset = int(np.frombuffer(bytes(index[entry : entry + 8]), dtype="<i8")[0])
        for raw, position in ((index, entry + 44), (data, offset + 4)):
            raw[position] = (raw[position] | 1) if flagged else (raw[position] & ~1)
        index_path.write_bytes(bytes(index))
        data_path.write_bytes(bytes(data))


def set_channel_rate(session_path: Path, channel: str, rate: float):
    """Give every segment of a channel the sampling frequency ``rate`` in its metadata."""
    for path in (session_path / f"{channel}.timd").rglob("*.tmet"):
        metadata = bytearray(path.read_bytes())
        metadata[8720:8728] = np.float64(rate).tobytes()  # TIME_SERIES_METADATA_SAMPLING_FREQUENCY_OFFSET
        path.write_bytes(bytes(metadata))


def rewrite_index_start_times(index_path: Path, change):
    """
    Replace the block start times of a .tidx by ``change(times)``, on their magnitudes: a writer stores a time
    offset by the recording time offset as a negative number, which keeps its sign.
    """
    raw = bytearray(index_path.read_bytes())
    entries = np.frombuffer(bytes(raw[1024:]), dtype="<i8").reshape(-1, 7).copy()  # 56-byte entries
    stored = entries[:, 1]  # TIME_SERIES_INDEX_START_TIME_OFFSET
    entries[:, 1] = np.sign(stored) * change(np.abs(stored))
    raw[1024:] = entries.tobytes()
    index_path.write_bytes(bytes(raw))


@pytest.fixture
def gapped_copy(tmp_path):
    destination = tmp_path / "gapped.mefd"
    shutil.copytree(GAPPED, destination)
    return destination


class TestREDDecoder:
    @pytest.mark.parametrize("use_numba", DECODERS)
    @pytest.mark.parametrize(
        "session_path, channel_name",
        [
            (GAPPED, "Cz"),
            (GAPPED, "Fp1"),
            (GAPPED, "EKG"),
            (LOSSY, "ch"),
            (TRUNCATED, "C3"),
            (TRUNCATED, "EKG"),
            (CONTIGUOUS, "A1"),
            (CONTIGUOUS, "A2"),
        ],
    )
    def test_matches_pymef(self, session_path, channel_name, use_numba):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            session = read_mef3_session(session_path)
        channel = next(channel for channel in session.channels if channel.name == channel_name)
        samples = read_all_blocks(channel, use_numba=use_numba)
        expected = EXPECTED[session_path.name]["channels"][channel_name]
        assert samples.dtype == np.int32
        assert samples.size == expected["number_of_samples"]
        assert digest(samples) == expected["sha256"]

    def test_lossy_blocks_are_scaled_and_detrended(self):
        """The fixture really exercises RED_unscale and RED_retrend: decoding is exact, the data is not the original."""
        with pytest.warns(UserWarning, match=r"2 blocks .* lossy options.*'ch' \(2\)\. The largest scale factor is"):
            recording = MEF3RecordingExtractor(LOSSY)
        blocks = recording.mef3_session.channels[0].blocks
        # meflib's encoder fell back to no scaling for block 0, which it marks with a negative factor that
        # RED_decode, like this decoder, ignores; block 1 is scaled. Both are detrended.
        assert blocks["scale_factor"][0] < 0 and blocks["scale_factor"][1] > 1
        assert (blocks["detrend_slope"] != 0).all()
        assert EXPECTED["lossy.mefd"]["original_sha256"] != EXPECTED["lossy.mefd"]["channels"]["ch"]["sha256"]

    @pytest.mark.parametrize("use_numba", DECODERS)
    @pytest.mark.parametrize(
        "start_shift, block_bytes",
        [
            pytest.param(0, 100, id="shorter-than-header"),
            pytest.param(0, 304, id="no-coded-data"),
            pytest.param(0, 10**9, id="past-the-buffer"),
            pytest.param(-(10**9), None, id="negative-start"),
        ],
    )
    def test_block_geometry_is_checked_before_decoding(self, use_numba, start_shift, block_bytes):
        """Both decoders refuse the same way, before reading; numba would otherwise read out of bounds."""
        session = read_mef3_session(GAPPED)
        blocks = session.channels[0].blocks[:1]
        data = np.fromfile(session.channels[0].data_paths[0], dtype=np.uint8)
        with pytest.raises(REDDecodeError, match="must lie within"):
            decode_red_blocks(
                data,
                block_starts=blocks["file_offset"] + start_shift,
                block_bytes=blocks["block_bytes"] if block_bytes is None else np.array([block_bytes]),
                difference_counts=blocks["difference_bytes"],
                sample_counts=blocks["number_of_samples"],
                scale_factors=blocks["scale_factor"],
                detrend_slopes=blocks["detrend_slope"],
                detrend_intercepts=blocks["detrend_intercept"],
                layout=MEF3_RED_LAYOUT,
                use_numba=use_numba,
            )

    @pytest.mark.parametrize("use_numba", DECODERS)
    def test_a_block_whose_index_and_header_agree_on_a_wrong_size(self, gapped_copy, use_numba, monkeypatch):
        """
        The review's case: the last block states 100 bytes in both its index entry and its RED header.

        The two agree, so reading the session succeeds; decoding the block raises REDDecodeError on both paths
        rather than an IndexError or an out-of-bounds read.
        """
        from neuroconv.datainterfaces.ecephys.mef import _mef3_reader

        segment_directory = gapped_copy / "Cz.timd" / "Cz-000001.segd"
        index_path, data_path = segment_directory / "Cz-000001.tidx", segment_directory / "Cz-000001.tdat"
        index, data = bytearray(index_path.read_bytes()), bytearray(data_path.read_bytes())
        entry = len(index) - 56
        offset = int(np.frombuffer(bytes(index[entry : entry + 8]), dtype="<i8")[0])
        index[entry + 28 : entry + 32] = np.uint32(100).tobytes()  # TIME_SERIES_INDEX_BLOCK_BYTES_OFFSET
        data[offset + 36 : offset + 40] = np.uint32(100).tobytes()  # RED_BLOCK_BLOCK_BYTES_OFFSET
        index_path.write_bytes(bytes(index))
        data_path.write_bytes(bytes(data))

        monkeypatch.setattr(
            _mef3_reader,
            "decode_red_blocks",
            lambda *args, **kwargs: decode_red_blocks(*args, **kwargs, use_numba=use_numba),
        )
        recording = MEF3RecordingExtractor(gapped_copy)
        with pytest.raises(REDDecodeError, match="stated to be 100 bytes"):
            recording.get_traces(segment_index=1, channel_ids=["Cz"])


class TestReader:
    def test_channels_are_in_acquisition_order(self):
        session = read_mef3_session(GAPPED)
        assert [channel.name for channel in session.channels] == ["Cz", "Fp1", "EKG"]
        assert MEF3RecordingInterface.get_available_channel_ids(GAPPED) == ["Cz", "Fp1", "EKG"]

    def test_a_gap_between_segments_starts_a_run(self):
        session = read_mef3_session(GAPPED)
        assert [(run.number_of_samples, run.start_time) for run in session.runs] == [
            (2000, START_TIME),
            (1000, START_TIME + 4_000_000),
        ]

    def test_block_times_have_the_recording_time_offset_removed(self):
        """pymef reports the true times; the fixture stores them offset, as negative numbers."""
        session = read_mef3_session(GAPPED)
        for channel in session.channels:
            expected = EXPECTED["gapped.mefd"]["channels"][channel.name]["block_start_times"]
            assert channel.blocks["start_time"].tolist() == expected

    def test_a_run_continues_across_contiguous_segments(self):
        """pymef flags each segment's first block; the run continues because that block starts where the last ended."""
        session = read_mef3_session(CONTIGUOUS)
        for channel, boundary in zip(session.channels, (6, 5)):
            assert channel.blocks["segment"][boundary - 1 : boundary + 1].tolist() == [0, 1]
            assert channel.blocks["discontinuity"][boundary]
            assert [(run.first_block, run.number_of_samples) for run in channel.runs] == [(0, 2400)]
        assert [(run.number_of_samples, run.start_time) for run in session.runs] == [(2400, START_TIME)]

    def test_clock_drift_is_not_a_gap(self):
        """ds004696's block 3 holds 10242 samples though the next block starts 10240 samples later."""
        session = read_mef3_session(TRUNCATED)
        blocks = session.channels[0].blocks
        assert blocks["number_of_samples"][3] == 10242
        assert round(np.diff(blocks["start_time"])[3] * 2048 / 1e6) == 10240
        assert len(session.runs) == 1
        assert session.runs[0].number_of_samples == 61442

    def test_a_flag_on_a_contiguous_block_does_not_start_a_run(self, gapped_copy):
        set_discontinuity_flag(gapped_copy, segment=0, block=4, flagged=True)
        session = read_mef3_session(gapped_copy)
        assert session.channels[0].blocks["discontinuity"][4]
        assert [run.number_of_samples for run in session.runs] == [2000, 1000]

    def test_an_unflagged_gap_of_more_than_a_second_starts_a_run(self, gapped_copy):
        set_discontinuity_flag(gapped_copy, segment=1, block=0, flagged=False)
        with pytest.warns(UserWarning, match=r"jump by more than a second at 1 blocks .*block 8 \(\+2\.000 s\)"):
            session = read_mef3_session(gapped_copy)
        assert [run.number_of_samples for run in session.runs] == [2000, 1000]

    def test_records_at_every_level(self):
        session = read_mef3_session(GAPPED)
        expected = sorted(EXPECTED["gapped.mefd"]["records"], key=lambda record: record["time"])
        assert [(record.time, record.record_type, record.channel) for record in session.records] == [
            (record["time"], record["type"], record["channel"]) for record in expected
        ]
        by_type = {record.record_type: record for record in session.records}
        assert by_type["Note"].label == "session note"
        assert (by_type["EDFA"].label, by_type["EDFA"].duration) == ("artifact", 1_500_000)
        assert (by_type["Seiz"].label, by_type["Seiz"].duration) == ("focal onset", 500_000)
        assert "marker_name_1=m1" in by_type["Seiz"].details
        # The body's own times are kept apart from the details, as true µUTC.
        assert by_type["Seiz"].times == {
            "earliest_onset": session.start_time + 300_000,
            "latest_offset": session.start_time + 800_000,
        }
        assert "earliest_onset" not in by_type["Seiz"].details and "latest_offset" not in by_type["Seiz"].details
        assert all(not record.times for record in session.records if record.record_type != "Seiz")
        assert (by_type["CSti"].label, by_type["CSti"].duration) == ("picture", 250_000)
        assert "task_type=naming" in by_type["CSti"].details
        assert by_type["ESti"].label == "biphasic"
        assert "amplitude=6.0 mA" in by_type["ESti"].details and "cathode=Fp1" in by_type["ESti"].details

    def test_gmt_offset(self):
        session = read_mef3_session(GAPPED)
        assert session.start_datetime == datetime(2000, 1, 1, tzinfo=timezone.utc)
        assert session.start_datetime.utcoffset().total_seconds() == 3600

    def test_encrypted_sessions_are_refused(self):
        with pytest.raises(ValueError, match="is encrypted"):
            read_mef3_session(ENCRYPTED)

    def test_channels_to_skip(self):
        session = read_mef3_session(GAPPED, channels_to_skip=["EKG"])
        assert [channel.name for channel in session.channels] == ["Cz", "Fp1"]
        with pytest.raises(ValueError, match=r"not in the session: \['Pz'\]"):
            read_mef3_session(GAPPED, channels_to_skip=["Pz"])

    def test_channels_at_different_rates_are_grouped(self, gapped_copy):
        set_channel_rate(gapped_copy, "EKG", 500.0)
        session = read_mef3_session(gapped_copy)
        assert [
            ([channel.name for channel in group.channels], group.sampling_frequency) for group in session.channel_groups
        ] == [(["Cz", "Fp1"], 1000.0), (["EKG"], 500.0)]
        # One recording holds one group.
        with pytest.raises(ValueError, match="fall into 2 groups"):
            session.runs
        with pytest.raises(ValueError, match="fall into 2 groups"):
            MEF3RecordingExtractor(gapped_copy)
        assert len(read_mef3_session(gapped_copy, channels_to_skip=["EKG"]).channels) == 2
        # Which channels to leave out, for each rate, without reading the indices.
        rates = MEF3RecordingInterface.get_channel_sampling_frequencies(gapped_copy)
        assert rates == {"Cz": 1000.0, "Fp1": 1000.0, "EKG": 500.0}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            only_ekg = read_mef3_session(gapped_copy, channels_to_skip=[name for name in rates if rates[name] != 500.0])
        assert [channel.name for channel in only_ekg.channels] == ["EKG"] and only_ekg.sampling_frequency == 500.0

    def test_not_a_session(self, tmp_path):
        with pytest.raises(ValueError, match="no .timd channel directories"):
            read_mef3_session(tmp_path)


class TestExtractor:
    def test_traces_across_blocks_and_runs(self):
        recording = MEF3RecordingExtractor(GAPPED)
        assert recording.get_num_segments() == 2
        assert recording.get_dtype() == np.int32
        session = recording.mef3_session
        full = np.stack([read_all_blocks(channel) for channel in session.channels], axis=1)
        np.testing.assert_array_equal(recording.get_traces(segment_index=0), full[:2000])
        np.testing.assert_array_equal(recording.get_traces(segment_index=1), full[2000:])
        # Windows that start and end inside blocks, span a block boundary, and select channels.
        np.testing.assert_array_equal(
            recording.get_traces(segment_index=0, start_frame=245, end_frame=1755), full[245:1755]
        )
        np.testing.assert_array_equal(
            recording.get_traces(segment_index=1, start_frame=1, end_frame=3, channel_ids=["EKG", "Cz"]),
            full[2001:2003][:, [2, 0]],
        )
        assert recording.get_traces(segment_index=0, start_frame=5, end_frame=5).shape == (0, 3)
        assert recording.get_start_time(segment_index=1) == 4.0

    @pytest.mark.parametrize(
        "start_frame, end_frame",
        [(0, 2400), (1499, 1501), (1200, 1800), (1500, 1501), (1499, 1500), (0, 1500), (1500, 2400)],
    )
    def test_traces_across_a_segment_boundary(self, start_frame, end_frame):
        """Windows within, ending at, starting at and straddling the boundary at sample 1500, against per-file decodes."""
        recording = MEF3RecordingExtractor(CONTIGUOUS)
        assert recording.get_num_segments() == 1
        # Decoded one data file at a time, independently of MEF3Channel.read_blocks and its per-segment split.
        reference = np.stack(
            [read_all_blocks(channel, use_numba=NUMBA_AVAILABLE) for channel in recording.mef3_session.channels],
            axis=1,
        )
        for column, channel in enumerate(recording.mef3_session.channels):
            assert digest(reference[:, column]) == EXPECTED["contiguous.mefd"]["channels"][channel.name]["sha256"]
        np.testing.assert_array_equal(
            recording.get_traces(start_frame=start_frame, end_frame=end_frame), reference[start_frame:end_frame]
        )
        np.testing.assert_array_equal(
            recording.get_traces(start_frame=start_frame, end_frame=end_frame, channel_ids=["A2"])[:, 0],
            reference[start_frame:end_frame, 1],
        )

    def test_gains_keep_their_sign_and_scale_to_microvolts(self):
        recording = MEF3RecordingExtractor(GAPPED)
        np.testing.assert_allclose(recording.get_channel_gains(), [0.5, -0.25, 1.0])
        assert list(recording.get_property("physical_unit")) == ["microvolts", "microvolts", "millivolts"]
        assert list(recording.get_property("reference")) == ["", "Cz", ""]

    def test_the_same_samples_with_and_without_threads(self, monkeypatch):
        from neuroconv.datainterfaces.ecephys.mef import _mef3_extractor

        recording = MEF3RecordingExtractor(TRUNCATED)
        threaded = recording.get_traces(start_frame=10000, end_frame=40000)
        monkeypatch.setattr(_mef3_extractor, "NUMBA_AVAILABLE", False)
        np.testing.assert_array_equal(recording.get_traces(start_frame=10000, end_frame=40000), threaded)


class TestInterface:
    def test_deidentified_start_time_is_used_with_a_warning(self):
        interface = MEF3RecordingInterface(folder_path=TRUNCATED)
        with pytest.warns(UserWarning, match="de-identified session") as record:
            metadata = interface.get_metadata()
        # Attributed to this file, not to neuroconv.
        assert record[0].filename == __file__
        assert metadata["NWBFile"]["session_start_time"] == datetime(1970, 1, 1, 14, 9, 9, 798945, tzinfo=timezone.utc)

    def test_conversion(self, tmp_path):
        interface = MEF3RecordingInterface(folder_path=GAPPED)
        metadata = interface.get_metadata()
        assert metadata["NWBFile"]["session_start_time"] == datetime(2000, 1, 1, tzinfo=timezone.utc)
        nwbfile_path = tmp_path / "gapped.nwb"
        interface.run_conversion(nwbfile_path=nwbfile_path, metadata=metadata, overwrite=True)

        full = np.stack([read_all_blocks(channel) for channel in interface.mef3_session.channels], axis=1)
        with NWBHDF5IO(nwbfile_path, "r") as io:
            nwbfile = io.read()
            series = sorted(nwbfile.acquisition.values(), key=lambda series: series.starting_time)
            assert [item.starting_time for item in series] == [0.0, 4.0]
            assert [item.rate for item in series] == [1000.0, 1000.0]
            assert series[0].data.dtype == np.int32
            np.testing.assert_array_equal(series[0].data[:], full[:2000])
            np.testing.assert_array_equal(series[1].data[:], full[2000:])
            # Microvolts per count, written as volts per count, with the sign the file stores.
            np.testing.assert_allclose(series[0].conversion * series[0].channel_conversion[:], [0.5e-6, -0.25e-6, 1e-6])
            assert list(nwbfile.electrodes["channel_name"][:]) == ["Cz", "Fp1", "EKG"]

            records = nwbfile.intervals["records"].to_dataframe()
            assert list(records["record_type"]) == ["Note", "Seiz", "ESti", "CSti", "EDFA"]
            np.testing.assert_allclose(records["start_time"], [0.1, 0.3, 1.234, 1.5, 2.5])
            np.testing.assert_allclose(records["stop_time"], [0.101, 0.8, 1.235, 1.75, 4.0])
            assert list(records["channel"]) == ["", "Cz", "Cz", "Cz", ""]
            assert set(records["source_file"]) == {"gapped.mefd"}
            # A time the record states is written on the same timeline as start_time, never as µUTC.
            seizure_details = records["details"][records["record_type"] == "Seiz"].iloc[0]
            assert "earliest_onset_s=0.300000" in seizure_details and "latest_offset_s=0.800000" in seizure_details
            assert not any(re.search(r"\d{10,}", details) for details in records["details"])

            runs = nwbfile.intervals["runs"].to_dataframe()
            assert list(runs["start_time"]) == [0.0, 4.0]
            assert list(runs["stop_time"]) == [2.0, 5.0]

    def test_tables_follow_alignment(self):
        interface = MEF3RecordingInterface(folder_path=GAPPED)
        # The base class shifts each segment's own times by the value given, so run 1 moves from 4.0 to 18.5.
        interface.set_aligned_segment_starting_times([10.0, 14.5])
        nwbfile = mock_NWBFile()
        interface.add_to_nwbfile(nwbfile=nwbfile, metadata=interface.get_metadata())
        series_starts = sorted(series.starting_time for series in nwbfile.acquisition.values())
        np.testing.assert_allclose(nwbfile.intervals["runs"]["start_time"][:], series_starts)
        np.testing.assert_allclose(series_starts, [10.0, 18.5])
        # The Note is 0.1 s into the first run.
        assert nwbfile.intervals["records"]["start_time"][0] == pytest.approx(10.1)
        # The seizure's stated times move with it.
        records = nwbfile.intervals["records"].to_dataframe()
        seizure_details = records["details"][records["record_type"] == "Seiz"].iloc[0]
        assert "earliest_onset_s=10.300000" in seizure_details and "latest_offset_s=10.800000" in seizure_details

    def test_tables_can_be_left_out(self):
        interface = MEF3RecordingInterface(folder_path=GAPPED)
        nwbfile = mock_NWBFile()
        interface.add_to_nwbfile(
            nwbfile=nwbfile, metadata=interface.get_metadata(), write_records=False, write_runs=False
        )
        assert "records" not in nwbfile.intervals and "runs" not in nwbfile.intervals

    def test_two_interfaces_over_one_session_write_its_records_once(self):
        first = MEF3RecordingInterface(folder_path=GAPPED, channels_to_skip=["EKG"], metadata_key="eeg")
        second = MEF3RecordingInterface(folder_path=GAPPED, channels_to_skip=["Cz", "Fp1"], metadata_key="ekg")
        nwbfile = mock_NWBFile()
        for interface in (first, second):
            metadata = interface.get_metadata()
            metadata["Ecephys"]["ElectricalSeries"][interface.metadata_key]["name"] = interface.metadata_key
            interface.add_to_nwbfile(nwbfile=nwbfile, metadata=metadata)
        assert len(nwbfile.intervals["records"]) == 5
        assert len(nwbfile.intervals["runs"]) == 2


def make_two_rate_session(session_path: Path, *, rate: float = 2000.0, later_by_microseconds: int = 3_000_000):
    """
    gapped.mefd with EKG at another rate, starting ``later_by_microseconds`` after Cz and Fp1, and holding a copy
    of Cz's channel-level records as its own.
    """
    ekg = session_path / "EKG.timd"
    set_channel_rate(session_path, "EKG", rate)
    for path in ekg.rglob("*.tidx"):
        rewrite_index_start_times(path, lambda times: times + later_by_microseconds)
    for suffix in (".ridx", ".rdat"):
        shutil.copy(session_path / "Cz.timd" / f"Cz{suffix}", ekg / f"EKG{suffix}")


class TestSessionAtSeveralRates:
    """One reader per rate, over complementary channels_to_skip, shares the session's timeline and records."""

    def test_skipped_channels_set_the_start_and_keep_their_records(self, gapped_copy):
        whole = read_mef3_session(GAPPED)
        make_two_rate_session(gapped_copy)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            main = read_mef3_session(gapped_copy, channels_to_skip=["EKG"])
            ekg = read_mef3_session(gapped_copy, channels_to_skip=["Cz", "Fp1"])

        assert main.start_time == ekg.start_time == whole.start_time == START_TIME
        assert ekg.runs[0].start_time == START_TIME + 3_000_000
        assert main.session_sampling_frequencies == ekg.session_sampling_frequencies == (1000.0, 2000.0)
        cz_records = [r for r in whole.records if r.channel == "Cz" and r.time]
        assert cz_records
        for session in (main, ekg):
            ekg_records = [r for r in session.records if r.channel == "EKG"]
            assert len(ekg_records) >= 1  # Cz's channel-level records, copied to EKG
            assert len(session.records) == len(main.records)
        recording = MEF3RecordingExtractor(gapped_copy, channels_to_skip=["Cz", "Fp1"])
        assert recording.get_start_time(segment_index=0) == pytest.approx(3.0)

    def test_an_unreadable_skipped_channel_is_left_out_with_a_warning(self, gapped_copy):
        for path in (gapped_copy / "EKG.timd").rglob("*.tidx"):
            path.write_bytes(b"not an index")
        with pytest.warns(UserWarning, match="skipped MEF 3 channel 'EKG'.*cannot be read"):
            session = read_mef3_session(gapped_copy, channels_to_skip=["EKG"])
        assert session.start_time == START_TIME

    def test_one_nwb_one_timeline(self, gapped_copy, tmp_path):
        """The two rates written through one ConverterPipe: one session_start_time, the later rate 3 s in."""
        from neuroconv import ConverterPipe

        single_rate_records = len(read_mef3_session(GAPPED).records)
        make_two_rate_session(gapped_copy)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            interfaces = {
                "mef3_1000Hz": MEF3RecordingInterface(
                    folder_path=gapped_copy, channels_to_skip=["EKG"], metadata_key="mef3_1000Hz"
                ),
                "mef3_2000Hz": MEF3RecordingInterface(
                    folder_path=gapped_copy, channels_to_skip=["Cz", "Fp1"], metadata_key="mef3_2000Hz"
                ),
            }
            pipe = ConverterPipe(data_interfaces=interfaces)
            metadata = pipe.get_metadata()
            for key in interfaces:
                metadata["Ecephys"]["ElectricalSeries"][key]["name"] = f"ElectricalSeries_{key[5:]}"
            nwbfile_path = tmp_path / "two_rates.nwb"
            pipe.run_conversion(nwbfile_path=nwbfile_path, metadata=metadata, overwrite=True)

        with NWBHDF5IO(nwbfile_path, "r") as io:
            nwbfile = io.read()
            assert nwbfile.session_start_time.astimezone(timezone.utc) == datetime(2000, 1, 1, tzinfo=timezone.utc)
            starts = {}
            for series in nwbfile.acquisition.values():
                starts.setdefault(series.rate, []).append(series.starting_time)
            assert min(starts[1000.0]) == 0.0
            assert min(starts[2000.0]) == pytest.approx(3.0)
            records = nwbfile.intervals["records"].to_dataframe()
            assert (records["channel"] == "EKG").sum() >= 1
            assert len(records) > single_rate_records
            runs = nwbfile.intervals["runs"].to_dataframe()
            assert set(runs["source_file"]) == {"gapped.mefd (1000 Hz)", "gapped.mefd (2000 Hz)"}

    def _per_rate_interfaces(self, session_path, rates_and_skips):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return {
                key: MEF3RecordingInterface(folder_path=session_path, channels_to_skip=skip, metadata_key=key)
                for key, skip in rates_and_skips
            }

    def test_rates_that_print_alike_keep_their_own_runs(self, gapped_copy):
        """1000 Hz and 1000.0001 Hz are two rates to the reader; their runs rows need two labels, or one is lost."""
        make_two_rate_session(gapped_copy, rate=1000.0001)
        interfaces = self._per_rate_interfaces(gapped_copy, [("main", ["EKG"]), ("ekg", ["Cz", "Fp1"])])
        nwbfile = mock_NWBFile()
        for interface in interfaces.values():
            interface._add_runs_to_nwbfile(nwbfile=nwbfile, run_start_times=interface._run_start_times())
        runs = nwbfile.intervals["runs"].to_dataframe()
        assert set(runs["source_file"]) == {"gapped.mefd (1000.0 Hz)", "gapped.mefd (1000.0001 Hz)"}
        assert len(runs) == sum(len(interface.mef3_session.runs) for interface in interfaces.values())

    def test_a_skipped_channel_without_a_rate_is_left_out(self, gapped_copy):
        set_channel_rate(gapped_copy, "EKG", -1.0)  # TIME_SERIES_METADATA_SAMPLING_FREQUENCY_NO_ENTRY
        with pytest.warns(UserWarning, match=r"skipped MEF 3 channel 'EKG'.*sampling frequency of -1\.0"):
            session = read_mef3_session(gapped_copy, channels_to_skip=["EKG"])
        assert session.session_sampling_frequencies == (1000.0,)
        assert session.start_time == START_TIME

    @pytest.mark.parametrize("damage", ["directory", "truncated"])
    def test_a_skipped_channel_whose_index_cannot_be_read_is_left_out(self, gapped_copy, damage):
        """An OSError (a directory where the index should be) or a truncated entry warns rather than raises."""
        for path in (gapped_copy / "EKG.timd").rglob("*.tidx"):
            if damage == "directory":
                path.unlink()
                path.mkdir()
            else:
                path.write_bytes(path.read_bytes()[:-1])
        with pytest.warns(UserWarning, match="skipped MEF 3 channel 'EKG'.*cannot be read"):
            session = read_mef3_session(gapped_copy, channels_to_skip=["EKG"])
        assert session.start_time == START_TIME

    def test_instant_records_span_the_same_whoever_writes_them(self, gapped_copy):
        """The records table is written once, by the first interface; its rows must not depend on which one."""
        make_two_rate_session(gapped_copy)
        interfaces = self._per_rate_interfaces(gapped_copy, [("main", ["EKG"]), ("ekg", ["Cz", "Fp1"])])
        tables = []
        for interface in interfaces.values():
            nwbfile = mock_NWBFile()
            interface._add_records_to_nwbfile(nwbfile=nwbfile, run_start_times=interface._run_start_times())
            tables.append(nwbfile.intervals["records"].to_dataframe().reset_index(drop=True))
        assert tables[0].equals(tables[1])
        instants = tables[0][tables[0]["duration_in_microseconds"] == 0]
        assert len(instants)
        np.testing.assert_allclose(instants["stop_time"] - instants["start_time"], 1 / 2000.0)


def shift_channel_segment(session_path: Path, channel: str, segment: int, by_microseconds: int):
    """Move every block of one segment of one channel by ``by_microseconds``, in its index."""
    stem = f"{channel}-{segment:06d}"
    index_path = session_path / f"{channel}.timd" / f"{stem}.segd" / f"{stem}.tidx"
    rewrite_index_start_times(index_path, lambda times: times + by_microseconds)


def make_slower_ekg(session_path: Path, *, later_by_microseconds: int = 3_000_000):
    """
    gapped.mefd with EKG at 500 Hz, its blocks re-timed to that rate (so its runs are 0-4 s and 8-10 s of its
    own time) and starting ``later_by_microseconds`` after Cz and Fp1.
    """
    set_channel_rate(session_path, "EKG", 500.0)
    index_paths = sorted((session_path / "EKG.timd").rglob("*.tidx"))
    first = abs(int(np.frombuffer(index_paths[0].read_bytes()[1032:1040], dtype="<i8")[0]))  # block 0's start
    for path in index_paths:
        rewrite_index_start_times(path, lambda times: first + 2 * (times - first) + later_by_microseconds)


def make_continuous_ekg(session_path: Path):
    """gapped.mefd with EKG recorded without the 2 s gap Cz and Fp1 have: its second segment follows its first."""
    shift_channel_segment(session_path, "EKG", segment=1, by_microseconds=-2_000_000)


def write_quietly(interface_or_pipe, nwbfile_path: Path, metadata: dict | None = None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        metadata = metadata or interface_or_pipe.get_metadata()
        interface_or_pipe.run_conversion(nwbfile_path=nwbfile_path, metadata=metadata, overwrite=True)


class TestChannelGroups:
    """A session whose channels cannot share one ElectricalSeries is written as one series per channel group."""

    def test_mixed_rates_are_written_as_one_series_per_rate(self, gapped_copy, tmp_path):
        make_slower_ekg(gapped_copy)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            session = read_mef3_session(gapped_copy)
            interface = MEF3RecordingInterface(folder_path=gapped_copy)
        full = {channel.name: read_all_blocks(channel) for channel in session.channels}
        assert interface.channel_ids == ["Cz", "Fp1", "EKG"]
        assert [part.metadata_key for part in interface.split_by_channel_group()] == [
            "mef3_recording_1000Hz",
            "mef3_recording_500Hz",
        ]
        nwbfile_path = tmp_path / "two_rates.nwb"
        write_quietly(interface, nwbfile_path)

        with NWBHDF5IO(nwbfile_path, "r") as io:
            nwbfile = io.read()
            assert nwbfile.session_start_time.astimezone(timezone.utc) == datetime(2000, 1, 1, tzinfo=timezone.utc)
            assert sorted(nwbfile.acquisition) == [
                "ElectricalSeries_1000Hz0",
                "ElectricalSeries_1000Hz1",
                "ElectricalSeries_500Hz0",
                "ElectricalSeries_500Hz1",
            ]
            series = nwbfile.acquisition
            # Each rate keeps its own runs, on the session's one timeline: EKG starts 3 s after Cz and Fp1.
            assert [(series[f"ElectricalSeries_1000Hz{run}"].starting_time) for run in (0, 1)] == [0.0, 4.0]
            assert [(series[f"ElectricalSeries_500Hz{run}"].starting_time) for run in (0, 1)] == [3.0, 11.0]
            assert {series[name].rate for name in series} == {1000.0, 500.0}
            main = np.stack([full["Cz"], full["Fp1"]], axis=1)
            np.testing.assert_array_equal(series["ElectricalSeries_1000Hz0"].data[:], main[:2000])
            np.testing.assert_array_equal(series["ElectricalSeries_1000Hz1"].data[:], main[2000:])
            np.testing.assert_array_equal(series["ElectricalSeries_500Hz0"].data[:, 0], full["EKG"][:2000])
            np.testing.assert_array_equal(series["ElectricalSeries_500Hz1"].data[:, 0], full["EKG"][2000:])
            # One electrodes table, each series pointing at its own rows.
            assert list(nwbfile.electrodes["channel_name"][:]) == ["Cz", "Fp1", "EKG"]
            assert list(series["ElectricalSeries_500Hz0"].electrodes.data[:]) == [2]
            np.testing.assert_allclose(
                series["ElectricalSeries_1000Hz0"].conversion
                * series["ElectricalSeries_1000Hz0"].channel_conversion[:],
                [0.5e-6, -0.25e-6],
            )

            records = nwbfile.intervals["records"].to_dataframe()
            assert len(records) == len(session.records)
            runs = nwbfile.intervals["runs"].to_dataframe()
            assert sorted(zip(runs["source_file"], runs["start_time"])) == [
                ("gapped.mefd (1000 Hz)", 0.0),
                ("gapped.mefd (1000 Hz)", 4.0),
                ("gapped.mefd (500 Hz)", 3.0),
                ("gapped.mefd (500 Hz)", 11.0),
            ]

    def test_channels_over_different_runs_at_one_rate(self, gapped_copy, tmp_path):
        make_continuous_ekg(gapped_copy)
        with pytest.warns(UserWarning, match=r"sampled at 1000 Hz were not all recorded over the same stretches"):
            session = read_mef3_session(gapped_copy)
        assert [
            ([channel.name for channel in group.channels], [run.number_of_samples for run in group.runs])
            for group in session.channel_groups
        ] == [(["Cz", "Fp1"], [2000, 1000]), (["EKG"], [3000])]
        full = {channel.name: read_all_blocks(channel) for channel in session.channels}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            interface = MEF3RecordingInterface(folder_path=gapped_copy)
        nwbfile_path = tmp_path / "two_layouts.nwb"
        write_quietly(interface, nwbfile_path)

        with NWBHDF5IO(nwbfile_path, "r") as io:
            nwbfile = io.read()
            series = nwbfile.acquisition
            assert sorted(series) == [
                "ElectricalSeries_1000Hz_00",
                "ElectricalSeries_1000Hz_01",
                "ElectricalSeries_1000Hz_1",
            ]
            assert [series[f"ElectricalSeries_1000Hz_0{run}"].starting_time for run in (0, 1)] == [0.0, 4.0]
            ekg = series["ElectricalSeries_1000Hz_1"]
            assert ekg.starting_time == 0.0 and ekg.rate == 1000.0
            np.testing.assert_array_equal(ekg.data[:, 0], full["EKG"])
            np.testing.assert_array_equal(
                series["ElectricalSeries_1000Hz_01"].data[:], np.stack([full["Cz"], full["Fp1"]], axis=1)[2000:]
            )
            # Only the group with gaps has runs to list.
            runs = nwbfile.intervals["runs"].to_dataframe()
            assert list(runs["source_file"]) == ["gapped.mefd (1000 Hz, group 0)"] * 2
            assert len(nwbfile.intervals["records"]) == len(session.records)

    def test_groups_through_a_converter_pipe_and_split_by_offset(self, gapped_copy, tmp_path):
        """split_by_offset gives the groups, so a caller that splits every recording by offset writes them all."""
        from neuroconv import ConverterPipe

        make_slower_ekg(gapped_copy)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            interface = MEF3RecordingInterface(folder_path=gapped_copy, metadata_key="mef3")
        parts = interface.split_by_offset()
        assert [(part.metadata_key, list(part.channel_ids)) for part in parts] == [
            ("mef3_1000Hz", ["Cz", "Fp1"]),
            ("mef3_500Hz", ["EKG"]),
        ]
        pipe = ConverterPipe(data_interfaces={part.metadata_key: part for part in parts})
        nwbfile_path = tmp_path / "pipe.nwb"
        write_quietly(pipe, nwbfile_path)
        with NWBHDF5IO(nwbfile_path, "r") as io:
            nwbfile = io.read()
            assert sorted(nwbfile.acquisition) == [
                "ElectricalSeries_1000Hz0",
                "ElectricalSeries_1000Hz1",
                "ElectricalSeries_500Hz0",
                "ElectricalSeries_500Hz1",
            ]
            assert nwbfile.acquisition["ElectricalSeries_500Hz0"].starting_time == pytest.approx(3.0)

    def test_old_metadata_format(self, gapped_copy):
        make_slower_ekg(gapped_copy)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            interface = MEF3RecordingInterface(folder_path=gapped_copy)
            metadata = interface.get_metadata(use_new_metadata_format=False)
            assert {"ElectricalSeries_1000Hz", "ElectricalSeries_500Hz"} <= set(metadata["Ecephys"])
            nwbfile = mock_NWBFile()
            interface.add_to_nwbfile(nwbfile=nwbfile, metadata=metadata)
        assert sorted(nwbfile.acquisition) == [
            "ElectricalSeries_1000Hz0",
            "ElectricalSeries_1000Hz1",
            "ElectricalSeries_500Hz0",
            "ElectricalSeries_500Hz1",
        ]

    def test_alignment_moves_every_group(self, gapped_copy):
        make_slower_ekg(gapped_copy)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            interface = MEF3RecordingInterface(folder_path=gapped_copy)
            interface.set_aligned_starting_time(10.0)
            nwbfile = mock_NWBFile()
            interface.add_to_nwbfile(nwbfile=nwbfile, metadata=interface.get_metadata())
        starts = {name: series.starting_time for name, series in nwbfile.acquisition.items()}
        assert starts == pytest.approx(
            {
                "ElectricalSeries_1000Hz0": 10.0,
                "ElectricalSeries_1000Hz1": 14.0,
                "ElectricalSeries_500Hz0": 13.0,
                "ElectricalSeries_500Hz1": 21.0,
            }
        )
        np.testing.assert_allclose(sorted(nwbfile.intervals["runs"]["start_time"][:]), sorted(starts.values()))
        # Per-run alignment needs one group's runs, so it is refused for the whole session.
        with pytest.raises(NotImplementedError, match="split_by_channel_group"):
            interface.set_aligned_segment_starting_times([0.0, 1.0])
        with pytest.raises(NotImplementedError, match="split_by_channel_group"):
            interface.get_timestamps()
        group = interface.split_by_channel_group()[1]
        assert [times[0] for times in group.get_timestamps()] == pytest.approx([13.0, 21.0])
        assert [times[0] for times in group.get_original_timestamps()] == pytest.approx([3.0, 11.0])

    def test_one_group_is_written_as_before(self, tmp_path):
        interface = MEF3RecordingInterface(folder_path=GAPPED)
        assert interface.split_by_channel_group() == [interface]
        assert interface.split_by_offset() == [interface]
        nwbfile = mock_NWBFile()
        interface.add_to_nwbfile(nwbfile=nwbfile, metadata=interface.get_metadata())
        assert sorted(nwbfile.acquisition) == ["ElectricalSeries0", "ElectricalSeries1"]
        assert set(nwbfile.intervals["runs"]["source_file"][:]) == {"gapped.mefd"}

    @pytest.mark.parametrize(
        "ekg_rate, labels, suffixes",
        [
            # One label under :g, distinct under .10g: the runs rows would merge.
            (1000.0001, ["1000.0", "1000.0001"], ["1000Hz", "1000.0001Hz"]),
            # One label under both: the series and metadata keys would collide.
            (1000.00000001, ["1000.0", "1000.00000001"], ["1000.0Hz", "1000.00000001Hz"]),
        ],
    )
    def test_rates_that_print_alike_keep_their_rows_and_series_apart(
        self, gapped_copy, tmp_path, ekg_rate, labels, suffixes
    ):
        """Two groups whose rates print alike are told apart by the exact rate, in the runs rows and the series."""
        set_channel_rate(gapped_copy, "EKG", ekg_rate)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            interface = MEF3RecordingInterface(folder_path=gapped_copy, metadata_key="mef3")
        groups = interface.split_by_channel_group()
        assert [group.metadata_key for group in groups] == [f"mef3_{suffix}" for suffix in suffixes]
        nwbfile_path = tmp_path / "close_rates.nwb"
        write_quietly(interface, nwbfile_path)
        with NWBHDF5IO(nwbfile_path, "r") as io:
            nwbfile = io.read()
            assert len(nwbfile.acquisition) == sum(group.recording_extractor.get_num_segments() for group in groups)
            runs = nwbfile.intervals["runs"].to_dataframe()
            assert sorted(set(runs["source_file"])) == [f"gapped.mefd ({label} Hz)" for label in labels]
            assert len(runs) == sum(len(group.mef3_session.runs) for group in groups)

    def test_a_group_split_by_offset_keeps_its_name(self, gapped_copy):
        """A group whose channels have different offsets splits further, under the group's own name."""
        make_slower_ekg(gapped_copy)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            interface = MEF3RecordingInterface(folder_path=gapped_copy)
            interface.split_by_channel_group()[0].recording_extractor.set_channel_offsets([0.0, 5.0])
            parts = interface.split_by_offset()
            assert [(part.metadata_key, list(part.channel_ids)) for part in parts] == [
                ("mef3_recording_1000Hz0", ["Cz"]),
                ("mef3_recording_1000Hz1", ["Fp1"]),
                ("mef3_recording_500Hz", ["EKG"]),
            ]
            metadata = parts[0].get_metadata()
        assert metadata["Ecephys"]["ElectricalSeries"]["mef3_recording_1000Hz0"]["name"] == "ElectricalSeries_1000Hz0"
        assert [part._split_series_name for part in parts] == [
            "ElectricalSeries_1000Hz0",
            "ElectricalSeries_1000Hz1",
            "ElectricalSeries_500Hz",
        ]


class TestEmptySegments:
    def test_an_empty_segment_is_left_out_with_a_warning(self, gapped_copy):
        whole = read_mef3_session(GAPPED)
        empty = gapped_copy / "Cz.timd" / "Cz-000002.segd"
        empty.mkdir()
        shutil.copy(gapped_copy / "Cz.timd" / "Cz-000001.segd" / "Cz-000001.tmet", empty / "Cz-000002.tmet")
        # An index of its universal header alone: no blocks.
        index = (gapped_copy / "Cz.timd" / "Cz-000001.segd" / "Cz-000001.tidx").read_bytes()[:1024]
        (empty / "Cz-000002.tidx").write_bytes(index)
        with pytest.warns(UserWarning, match=r"1 segments of the MEF 3 channel 'Cz.timd' hold no data file"):
            session = read_mef3_session(gapped_copy)
        assert len(session.channel_groups) == 1
        assert [run.number_of_samples for run in session.runs] == [run.number_of_samples for run in whole.runs]
        np.testing.assert_array_equal(read_all_blocks(session.channels[0]), read_all_blocks(whole.channels[0]))
        assert MEF3RecordingInterface.get_channel_sampling_frequencies(gapped_copy)["Cz"] == 1000.0

    def test_a_segment_with_blocks_and_no_data_file_is_refused(self, gapped_copy):
        (gapped_copy / "Cz.timd" / "Cz-000001.segd" / "Cz-000001.tdat").unlink()
        with pytest.raises(ValueError, match=r"holds 0 \.tdat files, not one"):
            read_mef3_session(gapped_copy)

    def test_a_channel_of_empty_segments_is_refused(self, gapped_copy):
        empty_channel(gapped_copy, "Cz", keep_data_files=False)
        with pytest.raises(ValueError, match="Every segment of the MEF 3 channel 'Cz.timd' is empty"):
            read_mef3_session(gapped_copy)
        assert len(read_mef3_session(gapped_copy, channels_to_skip=["Cz"]).channels) == 2


def empty_channel(session_path: Path, channel: str, keep_data_files: bool):
    """
    Leave every segment of a channel without a block: its indices cut to their universal header, and its data
    files cut to theirs, or removed, which makes each segment empty (see TestEmptySegments).
    """
    for segment_directory in (session_path / f"{channel}.timd").glob("*.segd"):
        for index_path in segment_directory.glob("*.tidx"):
            index_path.write_bytes(index_path.read_bytes()[:1024])
        for data_path in segment_directory.glob("*.tdat"):
            if keep_data_files:
                data_path.write_bytes(data_path.read_bytes()[:1024])
            else:
                data_path.unlink()


class TestEmptyChannels:
    """A channel that holds no samples is refused, listed by get_empty_channel_ids, and can be skipped."""

    @pytest.mark.parametrize("keep_data_files", [True, False], ids=["data files without blocks", "empty segments"])
    def test_listed_and_skipped_with_their_records_kept(self, gapped_copy, keep_data_files):
        whole = read_mef3_session(GAPPED)
        empty_channel(gapped_copy, "Cz", keep_data_files)
        assert MEF3RecordingInterface.get_empty_channel_ids(gapped_copy) == ["Cz"]
        message = "holds no samples. Leave it out with channels_to_skip; MEF3RecordingInterface.get_empty_channel_ids"
        with pytest.raises(ValueError, match=re.escape(message)):
            read_mef3_session(gapped_copy)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            session = read_mef3_session(gapped_copy, channels_to_skip=["Cz"])
        assert [channel.name for channel in session.channels] == ["Fp1", "EKG"]
        assert session.start_time == whole.start_time
        assert session.session_sampling_frequencies == whole.session_sampling_frequencies
        # The skipped channel's metadata is still read, so its records are kept and the session's are as before.
        assert session.records == whole.records
        assert any(record.channel == "Cz" for record in session.records)
        for channel in session.channels:
            whole_channel = next(other for other in whole.channels if other.name == channel.name)
            np.testing.assert_array_equal(read_all_blocks(channel), read_all_blocks(whole_channel))

    def test_an_interface_over_the_rest(self, gapped_copy, tmp_path):
        empty_channel(gapped_copy, "Fp1", keep_data_files=True)
        interface = MEF3RecordingInterface(
            folder_path=gapped_copy, channels_to_skip=MEF3RecordingInterface.get_empty_channel_ids(gapped_copy)
        )
        assert list(interface.channel_ids) == ["Cz", "EKG"]
        nwbfile_path = tmp_path / "without_fp1.nwb"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            interface.run_conversion(nwbfile_path=nwbfile_path, overwrite=True)
        with NWBHDF5IO(nwbfile_path, "r") as io:
            nwbfile = io.read()
            assert list(nwbfile.electrodes["channel_name"][:]) == ["Cz", "EKG"]

    def test_none_when_every_channel_holds_samples(self):
        for session_path in (GAPPED, LOSSY, CONTIGUOUS, TRUNCATED):
            assert MEF3RecordingInterface.get_empty_channel_ids(session_path) == []

    def test_every_channel_empty(self, gapped_copy):
        for channel in ("Cz", "Fp1", "EKG"):
            empty_channel(gapped_copy, channel, keep_data_files=True)
        assert MEF3RecordingInterface.get_empty_channel_ids(gapped_copy) == ["Cz", "EKG", "Fp1"]
        with pytest.raises(ValueError, match="channels_to_skip leaves no channels in the session"):
            read_mef3_session(gapped_copy, channels_to_skip=["Cz", "EKG", "Fp1"])

    def test_one_block_with_samples_is_not_empty(self, gapped_copy):
        """Every block but the last of one segment cut from the index: the channel still holds samples."""
        for index_path in sorted((gapped_copy / "Fp1.timd").rglob("*.tidx"))[1:]:
            index_path.write_bytes(index_path.read_bytes()[:1024])
        first = sorted((gapped_copy / "Fp1.timd").rglob("*.tidx"))[0]
        first.write_bytes(first.read_bytes()[: 1024 + 56])
        assert MEF3RecordingInterface.get_empty_channel_ids(gapped_copy) == []

    def test_an_unreadable_index_raises_rather_than_being_listed(self, gapped_copy):
        for index_path in (gapped_copy / "Fp1.timd").rglob("*.tidx"):
            index_path.write_bytes(index_path.read_bytes()[:-1])
        with pytest.raises(ValueError, match="truncated mid-entry"):
            MEF3RecordingInterface.get_empty_channel_ids(gapped_copy)

    def test_a_channel_without_segments_is_not_listed(self, gapped_copy):
        """Not a channel that holds no samples but a malformed one, which the reader refuses as such."""
        (gapped_copy / "Pz.timd").mkdir()
        assert MEF3RecordingInterface.get_empty_channel_ids(gapped_copy) == []
        with pytest.raises(ValueError, match=r"'.*Pz\.timd' holds no \.segd segment directories"):
            read_mef3_session(gapped_copy)

    def test_a_path_that_is_not_a_session_is_refused(self, tmp_path):
        """As read_mef3_session refuses it, rather than reading as a session with no empty channel."""
        with pytest.raises(ValueError, match="holds no .timd channel directories"):
            MEF3RecordingInterface.get_empty_channel_ids(tmp_path)
        with pytest.raises(ValueError, match="is not a directory"):
            MEF3RecordingInterface.get_empty_channel_ids(tmp_path / "missing.mefd")
        with pytest.raises(ValueError, match="holds no .timd channel directories"):
            MEF3RecordingInterface.get_channel_sampling_frequencies(tmp_path)

    def test_segments_with_only_their_metadata_file(self, gapped_copy):
        """A writer that opened the segments and never wrote: listed, and skipped with its records."""
        whole = read_mef3_session(GAPPED)
        for segment_directory in (gapped_copy / "Cz.timd").glob("*.segd"):
            for path in [*segment_directory.glob("*.tidx"), *segment_directory.glob("*.tdat")]:
                path.unlink()
        assert MEF3RecordingInterface.get_empty_channel_ids(gapped_copy) == ["Cz"]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            session = read_mef3_session(gapped_copy, channels_to_skip=["Cz"])
        assert session.records == whole.records
        assert session.start_time == whole.start_time

    def test_an_encrypted_session_without_its_password(self):
        """The indices are never encrypted, so a session's empty channels are listed without its password."""
        assert MEF3RecordingInterface.get_empty_channel_ids(ENCRYPTED) == []


class TestMEF3RecordingInterfaceContract(RecordingExtractorInterfaceTestMixin):
    data_interface_cls = MEF3RecordingInterface
    interface_kwargs = dict(folder_path=str(TRUNCATED))

    def check_extracted_metadata(self, metadata: dict):
        assert metadata["NWBFile"]["session_start_time"].year == 1970
        assert "mef3_recording" in metadata["Ecephys"]["ElectricalSeries"]


class TestGappedMEF3RecordingInterfaceContract(RecordingExtractorInterfaceTestMixin):
    data_interface_cls = MEF3RecordingInterface
    interface_kwargs = dict(folder_path=str(GAPPED))

    def check_extracted_metadata(self, metadata: dict):
        assert metadata["NWBFile"]["session_start_time"] == datetime(2000, 1, 1, tzinfo=timezone.utc)


class TestRecordBodyTimes:
    """The times a record body states are returned apart from its details."""

    @staticmethod
    def _cursor_body(trace_timestamp: int) -> bytes:
        name = b"cursor one".ljust(128, b"\0")
        return struct.pack("<qqqd", 7, trace_timestamp, 250, 1.5) + name

    def test_a_cursor_trace_time_is_a_body_time(self):
        label, details, duration, times = _record_body_fields("Curs", self._cursor_body(123_456_789_000_000))
        assert (label, duration) == ("cursor one", 0)
        assert details == "id_number=7; latency=250; value=1.5"
        assert times == {"trace_timestamp": 123_456_789_000_000}

    def test_a_time_with_no_entry_is_left_out(self):
        _, details, _, times = _record_body_fields("Curs", self._cursor_body(UUTC_NO_ENTRY))
        assert times == {} and "trace_timestamp" not in details

    def test_a_zero_filled_time_is_left_out(self):
        _, _, _, times = _record_body_fields("Curs", self._cursor_body(0))
        assert times == {}
        seizure = bytes(1312)  # a Seiz body its writer left zero-filled
        label, details, duration, times = _record_body_fields("Seiz", seizure)
        assert times == {} and "earliest_onset" not in details and "latest_offset" not in details

    def test_a_stated_time_before_the_session_is_not_written(self):
        interface = MEF3RecordingInterface(folder_path=GAPPED)
        seizure = next(record for record in interface.mef3_session.records if record.record_type == "Seiz")
        seizure.times["earliest_onset"] = interface.mef3_session.start_time - 1
        nwbfile = mock_NWBFile()
        interface.add_to_nwbfile(nwbfile=nwbfile, metadata=interface.get_metadata())
        records = nwbfile.intervals["records"].to_dataframe()
        details = records["details"][records["record_type"] == "Seiz"].iloc[0]
        assert "earliest_onset_s" not in details and "latest_offset_s=0.800000" in details
        assert not any(re.search(r"-\d", cell) for cell in records["details"])

    def test_types_without_body_times(self):
        assert _record_body_fields("Note", b"a note\0")[3] == {}
        assert _record_body_fields("Unkn", b"")[3] == {}


def move_recording_time_offset(session_path: Path, new_offset: int):
    """
    Give every segment the recording time offset ``new_offset`` and re-store every block start time against it,
    so the session reads exactly as before (the fixtures are not encrypted).
    """
    for metadata_path in session_path.rglob("*.tmet"):
        metadata = bytearray(metadata_path.read_bytes())
        old_offset = int(np.frombuffer(bytes(metadata[13312:13320]), dtype="<i8")[0])  # recording_time_offset
        metadata[13312:13320] = np.int64(new_offset).tobytes()
        metadata_path.write_bytes(bytes(metadata))
        # A stored magnitude is the true time minus the offset.
        rewrite_index_start_times(metadata_path.with_suffix(".tidx"), lambda times: times + (old_offset - new_offset))


def restore_record_times(session_path: Path, store, stored_record_times: dict | None = None):
    """
    Re-store the time of every record of a copy of the gapped fixture, and every time a Seiz body states, as
    ``store(true_time)``, given the true times the fixture holds. ``stored_record_times`` gives, by record type
    (each is used once in the fixture), a record's own stored time instead.
    """
    true_times = {
        (record.channel, record.record_type, record.label): record for record in read_mef3_session(GAPPED).records
    }
    for index_path in session_path.rglob("*.ridx"):
        index = bytearray(index_path.read_bytes())
        data = bytearray(index_path.with_suffix(".rdat").read_bytes())
        channel = "" if index_path.parent == session_path else index_path.parent.name.split(".")[0].split("-")[0]
        for entry in range(1024, len(index) - len(index) % 24, 24):
            offset = int(np.frombuffer(bytes(index[entry + 8 : entry + 16]), dtype="<i8")[0])
            record_type = bytes(data[offset + 4 : offset + 8]).decode()
            label, _, _, _ = _record_body_fields(record_type, bytes(data[offset + 24 :]))
            record = true_times[(channel, record_type, label)]
            stored = (stored_record_times or {}).get(record_type)
            stored = np.int64(store(record.time) if stored is None else stored).tobytes()
            index[entry + 16 : entry + 24] = stored
            data[offset + 16 : offset + 24] = stored
            for name, body_offset in (("earliest_onset", 0), ("latest_offset", 8)):
                if name in record.times:
                    position = offset + 24 + body_offset
                    data[position : position + 8] = np.int64(store(record.times[name])).tobytes()
        index_path.write_bytes(bytes(index))
        index_path.with_suffix(".rdat").write_bytes(bytes(data))


class TestRecordsOffsetByTheSessionStart:
    """
    A writer may store its records offset by the session's start rather than by the recording time offset its
    blocks are stored against. Read as MEF 3 has them, with an offset decades before the session, every record
    would fall decades before it too, and on the session timeline each would be minus the session's start.
    """

    FAR_OFFSET = START_TIME - 25 * 365 * 86_400_000_000  # an offset 25 years before the session

    def expected(self):
        return [
            (record.time, record.record_type, record.channel, record.times)
            for record in read_mef3_session(GAPPED).records
        ]

    def test_they_are_read_from_the_session_start(self, gapped_copy):
        expected = self.expected()
        move_recording_time_offset(gapped_copy, self.FAR_OFFSET)
        restore_record_times(gapped_copy, lambda true_time: -(true_time - START_TIME))
        with pytest.warns(UserWarning, match=r"5 records .* offset by the session's start"):
            session = read_mef3_session(gapped_copy)
        assert session.start_time == START_TIME
        assert [(r.time, r.record_type, r.channel, r.times) for r in session.records] == expected

    def test_a_record_stored_as_0_is_at_the_session_start(self, gapped_copy):
        move_recording_time_offset(gapped_copy, self.FAR_OFFSET)
        restore_record_times(
            gapped_copy, lambda true_time: -(true_time - START_TIME) if true_time != START_TIME + 100_000 else 0
        )
        with pytest.warns(UserWarning, match="offset by the session's start"):
            session = read_mef3_session(gapped_copy)
        note = next(record for record in session.records if record.record_type == "Note")
        assert note.time == START_TIME

    def test_records_stored_against_the_recording_time_offset_are_read_with_it(self, gapped_copy):
        expected = self.expected()
        move_recording_time_offset(gapped_copy, self.FAR_OFFSET)
        restore_record_times(gapped_copy, lambda true_time: -(true_time - self.FAR_OFFSET))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            session = read_mef3_session(gapped_copy)
        assert [(r.time, r.record_type, r.channel, r.times) for r in session.records] == expected

    def test_a_file_that_does_not_fit_the_session_from_its_start_is_read_as_stored(self, gapped_copy):
        # One record 10 s past the session's end: read from the start, not every record falls within the session.
        move_recording_time_offset(gapped_copy, self.FAR_OFFSET)
        late = START_TIME + 2_500_000  # the EDFA, a session-level record
        restore_record_times(
            gapped_copy, lambda true_time: -(true_time - START_TIME) - (10_000_000 if true_time == late else 0)
        )
        # The decision is each record file's own: the channel and segment files still fit, and are read from it.
        with pytest.warns(UserWarning, match=r"^3 records .* offset by the session's start"):
            session = read_mef3_session(gapped_copy)
        session_level = [record for record in session.records if not record.channel]
        assert len(session_level) == 2 and all(record.time < START_TIME for record in session_level)

    def test_on_the_session_timeline(self, gapped_copy):
        move_recording_time_offset(gapped_copy, self.FAR_OFFSET)
        restore_record_times(gapped_copy, lambda true_time: -(true_time - START_TIME))
        with pytest.warns(UserWarning, match="offset by the session's start"):
            interface = MEF3RecordingInterface(folder_path=gapped_copy)
        nwbfile = mock_NWBFile()
        interface.add_to_nwbfile(nwbfile=nwbfile, metadata=interface.get_metadata())
        records = nwbfile.intervals["records"].to_dataframe()
        np.testing.assert_allclose(records["start_time"], [0.1, 0.3, 1.234, 1.5, 2.5])
        seizure_details = records["details"][records["record_type"] == "Seiz"].iloc[0]
        assert "earliest_onset_s=0.300000" in seizure_details and "latest_offset_s=0.800000" in seizure_details

    def test_complementary_channel_skips_read_the_same_records(self, gapped_copy):
        """The span's end is taken over skipped channels too, so which channels are kept cannot change the records."""
        move_recording_time_offset(gapped_copy, self.FAR_OFFSET)
        # EKG's second segment 5 s later: EKG runs to 10 s, Cz and Fp1 to 5 s.
        rewrite_index_start_times(
            gapped_copy / "EKG.timd" / "EKG-000001.segd" / "EKG-000001.tidx", lambda t: t + 5_000_000
        )
        late_edfa = -9_500_000  # the session-level EDFA, 9.5 s after the start: within EKG's span only
        restore_record_times(gapped_copy, lambda true_time: -(true_time - START_TIME), {"EDFA": late_edfa})
        read = {}
        for skip in (["EKG"], ["Cz", "Fp1"]):
            with pytest.warns(UserWarning, match=r"^5 records .* offset by the session's start"):
                session = read_mef3_session(gapped_copy, channels_to_skip=skip)
            read[tuple(skip)] = [(r.time, r.record_type, r.channel, r.times) for r in session.records]
        assert read[("EKG",)] == read[("Cz", "Fp1")]
        edfa = next(record for record in read[("EKG",)] if record[1] == "EDFA")
        assert edfa[0] == START_TIME + 9_500_000

    def test_records_just_before_the_first_sample_stay_as_stored(self, gapped_copy):
        """With the offset 1 s before the first sample, as the fixtures have it, the two readings are a second apart:
        a Seiz stored at its true time 0.5 s before the session stays there, and nothing warns."""
        fixture_offset = START_TIME - 1_000_000
        restore_record_times(
            gapped_copy,
            lambda true_time: -(true_time - fixture_offset),
            {"Seiz": -((START_TIME - 500_000) - fixture_offset)},
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            session = read_mef3_session(gapped_copy)
        seizure = next(record for record in session.records if record.record_type == "Seiz")
        assert seizure.time == START_TIME - 500_000
        assert seizure.times == {"earliest_onset": START_TIME + 300_000, "latest_offset": START_TIME + 800_000}

    def test_a_time_stored_as_no_entry_does_not_move_a_file(self, gapped_copy):
        expected = {r.record_type: r.times for r in read_mef3_session(GAPPED).records}
        move_recording_time_offset(gapped_copy, self.FAR_OFFSET)
        restore_record_times(gapped_copy, lambda true_time: -(true_time - self.FAR_OFFSET), {"Seiz": UUTC_NO_ENTRY})
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            session = read_mef3_session(gapped_copy)
        seizure = next(record for record in session.records if record.record_type == "Seiz")
        assert seizure.time == UUTC_NO_ENTRY  # no time stated, returned as meflib returns it
        assert seizure.times == expected["Seiz"]

    def test_a_record_with_no_time_is_left_out_of_the_table(self, gapped_copy):
        move_recording_time_offset(gapped_copy, self.FAR_OFFSET)
        restore_record_times(gapped_copy, lambda true_time: -(true_time - self.FAR_OFFSET), {"Seiz": UUTC_NO_ENTRY})
        interface = MEF3RecordingInterface(folder_path=gapped_copy)
        nwbfile = mock_NWBFile()
        with pytest.warns(UserWarning, match=r"^1 records .* state no time"):
            interface.add_to_nwbfile(nwbfile=nwbfile, metadata=interface.get_metadata())
        records = nwbfile.intervals["records"].to_dataframe()
        assert list(records["record_type"]) == ["Note", "ESti", "CSti", "EDFA"]
        np.testing.assert_allclose(records["start_time"], [0.1, 1.234, 1.5, 2.5])

    def test_a_file_of_records_stored_as_0_stays_on_an_offset_near_the_start(self, gapped_copy):
        """A record stored as 0 is at the recording time offset, as meflib's writer means it: 1 s before the session
        with the fixtures' offset. The two readings are then a second apart, so the file is not moved."""
        fixture_offset = START_TIME - 1_000_000
        restore_record_times(gapped_copy, lambda true_time: -(true_time - fixture_offset), {"Seiz": 0})
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            session = read_mef3_session(gapped_copy)
        seizure = next(record for record in session.records if record.record_type == "Seiz")
        assert seizure.time == fixture_offset
        assert seizure.times == {"earliest_onset": START_TIME + 300_000, "latest_offset": START_TIME + 800_000}
        interface = MEF3RecordingInterface(folder_path=gapped_copy)
        nwbfile = mock_NWBFile()
        interface.add_to_nwbfile(nwbfile=nwbfile, metadata=interface.get_metadata())
        records = nwbfile.intervals["records"].to_dataframe()
        seizure_row = records[records["record_type"] == "Seiz"].iloc[0]
        assert seizure_row["start_time"] == pytest.approx(-1.0)
        assert "latest_offset_s=0.800000" in seizure_row["details"]

    def test_no_entry_is_returned_as_it_is(self):
        assert _true_time(UUTC_NO_ENTRY, START_TIME) == UUTC_NO_ENTRY
        assert _true_time(np.array([UUTC_NO_ENTRY, -5], dtype=np.int64), 10).tolist() == [UUTC_NO_ENTRY, 15]


class TestConcatenatedRuns:
    """A session's runs read as one segment with a time vector, and written as one series with timestamps."""

    @staticmethod
    def expected_times():
        # gapped.mefd: 2000 samples from 0 s, then 1000 from 4 s, at 1000 Hz.
        return np.concatenate([np.arange(2000) / 1000.0, 4.0 + np.arange(1000) / 1000.0])

    def test_the_extractor_reads_the_runs_as_one_segment(self):
        recording = MEF3RecordingExtractor(GAPPED, concatenate_runs=True)
        assert recording.get_num_segments() == 1
        assert recording.has_time_vector(segment_index=0)
        full = np.stack([read_all_blocks(channel) for channel in recording.mef3_session.channels], axis=1)
        np.testing.assert_array_equal(recording.get_traces(), full)
        np.testing.assert_allclose(recording.get_times(segment_index=0), self.expected_times(), rtol=0, atol=1e-12)
        # Windows within a run, ending at, starting at and straddling the boundary between the runs.
        for start_frame, end_frame in [(245, 1755), (1990, 2000), (2000, 2010), (1990, 2010), (0, 3000)]:
            np.testing.assert_array_equal(
                recording.get_traces(start_frame=start_frame, end_frame=end_frame), full[start_frame:end_frame]
            )
        np.testing.assert_array_equal(
            recording.get_traces(start_frame=1999, end_frame=2002, channel_ids=["EKG", "Cz"]),
            full[1999:2002][:, [2, 0]],
        )
        assert recording.to_dict()["kwargs"]["concatenate_runs"] is True

    def test_a_session_with_one_run_keeps_its_rate(self):
        recording = MEF3RecordingExtractor(CONTIGUOUS, concatenate_runs=True)
        assert recording.get_num_segments() == 1
        assert not recording.has_time_vector(segment_index=0)

    def test_the_interface_writes_one_series_with_timestamps(self, tmp_path):
        interface = MEF3RecordingInterface(folder_path=GAPPED, concatenate_runs_above=1)
        full = np.stack([read_all_blocks(channel) for channel in interface.mef3_session.channels], axis=1)
        nwbfile_path = tmp_path / "concatenated.nwb"
        write_quietly(interface, nwbfile_path)
        with NWBHDF5IO(nwbfile_path, "r") as io:
            nwbfile = io.read()
            assert list(nwbfile.acquisition) == ["ElectricalSeries"]
            series = nwbfile.acquisition["ElectricalSeries"]
            assert series.rate is None
            np.testing.assert_allclose(series.timestamps[:], self.expected_times(), rtol=0, atol=1e-12)
            np.testing.assert_array_equal(series.data[:], full)
            np.testing.assert_allclose(series.conversion * series.channel_conversion[:], [0.5e-6, -0.25e-6, 1e-6])
            # The tables are those of one series per run.
            runs = nwbfile.intervals["runs"].to_dataframe()
            assert list(runs["start_time"]) == [0.0, 4.0]
            assert list(runs["stop_time"]) == [2.0, 5.0]
            records = nwbfile.intervals["records"].to_dataframe()
            np.testing.assert_allclose(records["start_time"], [0.1, 0.3, 1.234, 1.5, 2.5])

    def test_a_group_with_no_more_runs_than_the_limit_keeps_one_series_per_run(self):
        interface = MEF3RecordingInterface(folder_path=GAPPED, concatenate_runs_above=2)
        assert interface.recording_extractor.get_num_segments() == 2
        nwbfile = mock_NWBFile()
        interface.add_to_nwbfile(nwbfile=nwbfile, metadata=interface.get_metadata())
        assert sorted(series.starting_time for series in nwbfile.acquisition.values()) == [0.0, 4.0]

    def test_each_channel_group_is_decided_on_its_own_runs(self, gapped_copy, tmp_path):
        make_continuous_ekg(gapped_copy)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            session = read_mef3_session(gapped_copy)
            interface = MEF3RecordingInterface(folder_path=gapped_copy, concatenate_runs_above=1)
        full = {channel.name: read_all_blocks(channel) for channel in session.channels}
        nwbfile_path = tmp_path / "groups.nwb"
        write_quietly(interface, nwbfile_path)
        with NWBHDF5IO(nwbfile_path, "r") as io:
            nwbfile = io.read()
            series = nwbfile.acquisition
            assert sorted(series) == ["ElectricalSeries_1000Hz_0", "ElectricalSeries_1000Hz_1"]
            gapped_group = series["ElectricalSeries_1000Hz_0"]
            np.testing.assert_allclose(gapped_group.timestamps[:], self.expected_times(), rtol=0, atol=1e-12)
            np.testing.assert_array_equal(gapped_group.data[:], np.stack([full["Cz"], full["Fp1"]], axis=1))
            # The continuous group has one run, so it keeps a starting time and rate.
            ekg = series["ElectricalSeries_1000Hz_1"]
            assert ekg.timestamps is None and ekg.starting_time == 0.0 and ekg.rate == 1000.0
            runs = nwbfile.intervals["runs"].to_dataframe()
            assert list(runs["source_file"]) == ["gapped.mefd (1000 Hz, group 0)"] * 2
            assert list(runs["start_time"]) == [0.0, 4.0]

    def test_tables_follow_alignment(self):
        interface = MEF3RecordingInterface(folder_path=GAPPED, concatenate_runs_above=1)
        interface.set_aligned_starting_time(10.0)
        nwbfile = mock_NWBFile()
        interface.add_to_nwbfile(nwbfile=nwbfile, metadata=interface.get_metadata())
        series = nwbfile.acquisition["ElectricalSeries"]
        np.testing.assert_allclose(series.timestamps[[0, 2000]], [10.0, 14.0])
        np.testing.assert_allclose(nwbfile.intervals["runs"]["start_time"][:], [10.0, 14.0])
        assert nwbfile.intervals["records"]["start_time"][0] == pytest.approx(10.1)

    def test_a_stub_of_concatenated_runs(self):
        """A stub slices the segment by frames, which rebuilds it from its times without the time vector."""
        interface = MEF3RecordingInterface(folder_path=GAPPED, concatenate_runs_above=1)
        nwbfile = mock_NWBFile()
        interface.add_to_nwbfile(nwbfile=nwbfile, metadata=interface.get_metadata(), stub_test=True)
        series = nwbfile.acquisition["ElectricalSeries"]
        # The first 100 samples lie in the first run, so the stub is placed by its start and rate.
        assert series.data.maxshape == (100, 3)
        assert (series.starting_time, series.rate, series.timestamps) == (0.0, 1000.0, None)

    @staticmethod
    def make_overlapping_runs(session_path: Path):
        """gapped.mefd with every channel's second segment moved 3 s earlier: runs at 0 s for 2 s and at 1 s for 1 s."""
        for channel in ("Cz", "Fp1", "EKG"):
            shift_channel_segment(session_path, channel, segment=1, by_microseconds=-3_000_000)

    def test_runs_that_go_back_in_time_stay_one_series_each(self, gapped_copy):
        self.make_overlapping_runs(gapped_copy)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert [run.start_time - START_TIME for run in read_mef3_session(gapped_copy).runs] == [0, 1_000_000]
        with pytest.warns(UserWarning, match=r"not concatenated: a run starts before the previous one ends"):
            interface = MEF3RecordingInterface(folder_path=gapped_copy, concatenate_runs_above=1)
        assert interface.recording_extractor.get_num_segments() == 2
        nwbfile = mock_NWBFile()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            interface.add_to_nwbfile(nwbfile=nwbfile, metadata=interface.get_metadata())
        assert sorted(series.starting_time for series in nwbfile.acquisition.values()) == [0.0, 1.0]
        assert all(series.timestamps is None for series in nwbfile.acquisition.values())
        with pytest.raises(ValueError, match=r"cannot be concatenated: a run starts before the previous one ends"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                MEF3RecordingExtractor(gapped_copy, concatenate_runs=True)

    def test_records_are_placed_in_the_last_run_that_starts_before_them(self):
        """Bisection over ascending starts, and a check of every run when a run starts before the previous one,
        both pick the run master's scan picked: the last that starts at or before the time, else the first."""
        interface = MEF3RecordingInterface(folder_path=GAPPED)
        start = interface.mef3_session.start_time
        aligned = np.array([10.0, 30.0])  # the runs' starts as written, moved from 0 s and 4 s
        for starts_ascend, utc in ((True, [0, 4_000_000]), (False, [4_000_000, 1_000_000])):
            utc = np.array(utc, dtype=np.int64) + start
            for offset in (-500_000, 0, 1_500_000, 4_000_000, 4_250_000):
                time = start + offset
                at_or_before = [index for index, run_start in enumerate(utc) if time >= run_start]
                expected_run = at_or_before[-1] if at_or_before else 0
                expected = (time - start) / 1e6 + aligned[expected_run] - (utc[expected_run] - start) / 1e6
                assert interface._session_time(time, aligned, utc, starts_ascend) == pytest.approx(expected)
