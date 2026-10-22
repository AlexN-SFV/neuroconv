"""Tests for the MEF3RecordingInterface, its reader and the RED decoder.

The sessions in ``mef3_fixtures/`` were written, or cut from ds004696, by ``build_mef3_fixtures.py``, which
also stored in ``expected.json`` what pymef reads from them. pymef is not needed to run these tests.
"""

import hashlib
import json
import shutil
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
from neuroconv.datainterfaces.ecephys.mef._mef3_reader import read_mef3_session
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
    def test_a_block_cut_short_raises(self, use_numba):
        session = read_mef3_session(GAPPED)
        blocks = session.channels[0].blocks[:1]
        data = np.fromfile(session.channels[0].data_paths[0], dtype=np.uint8)
        with pytest.raises(REDDecodeError, match="cannot be decoded"):
            decode_red_blocks(
                data,
                block_starts=blocks["file_offset"],
                block_bytes=np.array([304]),  # the header alone: no coded data for the samples it claims
                difference_counts=blocks["difference_bytes"],
                sample_counts=blocks["number_of_samples"],
                scale_factors=blocks["scale_factor"],
                detrend_slopes=blocks["detrend_slope"],
                detrend_intercepts=blocks["detrend_intercept"],
                layout=MEF3_RED_LAYOUT,
                use_numba=use_numba,
            )


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

    def test_channels_at_different_rates_are_refused(self, gapped_copy):
        for path in (gapped_copy / "EKG.timd").rglob("*.tmet"):
            metadata = bytearray(path.read_bytes())
            metadata[8720:8728] = np.float64(500.0).tobytes()  # TIME_SERIES_METADATA_SAMPLING_FREQUENCY_OFFSET
            path.write_bytes(bytes(metadata))
        with pytest.raises(ValueError, match="2 different rates"):
            read_mef3_session(gapped_copy)
        assert len(read_mef3_session(gapped_copy, channels_to_skip=["EKG"]).channels) == 2

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
