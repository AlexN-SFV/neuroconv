"""Tests for the Blackrock recording and sorting interfaces, on small file sets written here.

The writer below lays out .nsX and .nev files from the field layouts Blackrock's readers use (NPMK's openNSx and
openNEV, brpylib), so each case the interfaces handle is pinned without GIN data: pauses, clock resets, PTP
packets out of order or missing, FileSpec 2.1 scaling from the .nev, repeated labels, comments with leftover
bytes and file names with dots. The GIN files themselves are covered in ``tests/test_on_data``.
"""

import struct
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from pynwb import NWBHDF5IO
from pynwb.testing.mock.file import mock_NWBFile

from neuroconv import ConverterPipe
from neuroconv.datainterfaces import (
    BlackrockRecordingInterface,
    BlackrockSortingInterface,
)
from neuroconv.datainterfaces.ecephys.blackrock import _blackrock_file_set
from neuroconv.tools.testing.data_interface_mixins import (
    RecordingExtractorInterfaceTestMixin,
)

TIME_ORIGIN = (2020, 1, 3, 2, 4, 5, 6, 7)  # year, month, weekday, day, hour, minute, second, millisecond
TIME_ORIGIN_DATETIME = datetime(2020, 1, 2, 4, 5, 6, 7000, tzinfo=timezone.utc)
NEURAL = dict(units="uV", min_digital=-32764, max_digital=32764, min_analog=-8191, max_analog=8191)
ANALOG = dict(units="mV", min_digital=-32767, max_digital=32767, min_analog=-5000, max_analog=5000)


def channel(electrode_id, label, kind=NEURAL):
    return dict(electrode_id=electrode_id, label=label, **kind)


def _text(value: str, size: int) -> bytes:
    return value.encode("latin-1").ljust(size, b"\x00")[:size]


def write_nsx(path, *, channels, blocks, period=30, spec="2.3", comment="", resolution=30_000):
    """A FileSpec 2.2, 2.3 or 3.0 .nsX file of timestamped data blocks: ``blocks`` is [(clock value, samples)]."""
    major, minor = (int(part) for part in spec.split("."))
    file_id = b"BRSMPGRP" if major >= 3 else b"NEURALCD"
    extended = b"".join(
        struct.pack(
            "<2sH16sBBhhhh16sIIHIIH",
            b"CC",
            c["electrode_id"],
            _text(c["label"], 16),
            1,
            c["electrode_id"] % 256,
            c["min_digital"],
            c["max_digital"],
            c["min_analog"],
            c["max_analog"],
            _text(c["units"], 16),
            300,
            1,
            1,
            7_500_000,
            3,
            1,
        )
        for c in channels
    )
    basic_size = 314
    basic = (
        file_id
        + struct.pack("<BBI", major, minor, basic_size + len(extended))
        + _text("1 kS/s", 16)
        + _text(comment, 256)
        + struct.pack("<II8HI", period, resolution, *TIME_ORIGIN, len(channels))
    )
    assert len(basic) == basic_size
    timestamp_format = "<BQI" if major >= 3 else "<BII"
    data = b"".join(
        struct.pack(timestamp_format, 1, timestamp, len(samples)) + np.asarray(samples, dtype="<i2").tobytes()
        for timestamp, samples in blocks
    )
    Path(path).write_bytes(basic + extended + data)


def write_ptp_nsx(path, *, channels, timestamps, samples, period=30):
    """A PTP .nsX file (FileSpec 3.0, nanosecond clock): one packet per sample, each with its own timestamp."""
    write_nsx(path, channels=channels, blocks=[], period=period, spec="3.0", resolution=1_000_000_000)
    packets = b"".join(
        struct.pack("<BQI", 1, int(timestamp), 1) + np.asarray(row, dtype="<i2").tobytes()
        for timestamp, row in zip(timestamps, samples)
    )
    with open(path, "ab") as file:
        file.write(packets)


def write_v21_nsx(path, *, electrode_ids, samples, period=30):
    """A FileSpec 2.1 .nsX file: a 32-byte header, the electrode ids, then every sample, with no timestamps."""
    header = b"NEURALSG" + _text("1 kS/s", 16) + struct.pack("<II", period, len(electrode_ids))
    ids = struct.pack(f"<{len(electrode_ids)}I", *electrode_ids)
    Path(path).write_bytes(header + ids + np.asarray(samples, dtype="<i2").tobytes())


def digital(timestamp, word, reason=1):
    return ("digital", timestamp, reason, word)


def comment(timestamp, raw: bytes, char_set=0):
    return ("comment", timestamp, char_set, raw)


def spike(timestamp, electrode, unit=0):
    return ("spike", timestamp, electrode, unit)


def write_nev(
    path,
    *,
    packets,
    electrodes=((1, 250),),
    spec="2.3",
    resolution=30_000,
    application="File Dialog v7.6.1",
    time_origin=TIME_ORIGIN,
):
    """A .nev file of digital input, comment and spike packets, with one NEUEVWAV header per electrode."""
    major, minor = (int(part) for part in spec.split("."))
    packet_size = 64
    timestamp_format = "<QH" if major >= 3 else "<IH"
    extended = b""
    for electrode_id, factor in electrodes:
        tail = struct.pack("<BBH8s", 0, 2, 4, b"") if spec in ("2.3", "3.0") else struct.pack("<BB10s", 0, 2, b"")
        extended += b"NEUEVWAV" + struct.pack("<HBBHHhh", electrode_id, 1, 1, factor, 0, 0, 0) + tail
    header_size = 336 + len(extended)
    basic = (
        (b"BREVENTS" if major >= 3 else b"NEURALEV")
        # Flag bit 0, 16-bit waveforms, as every GIN file sets it; neo fails on a .nev without it (it indexes a
        # dict with the array of bytes-per-waveform values).
        + struct.pack("<BBHIIII", major, minor, 1, header_size, packet_size, resolution, 30_000)
        + struct.pack("<8H", *time_origin)
        + _text(application, 32)
        + _text("", 256)
        + struct.pack("<I", len(electrodes))
    )
    assert len(basic) == 336
    body = b""
    for kind, timestamp, first, second in packets:
        if kind == "digital":
            payload = struct.pack("<BBH", first, 0, second)
            packet_id = 0
        elif kind == "comment":
            payload = struct.pack("<BBI", first, 0, 0) + second
            packet_id = 0xFFFF
        else:
            payload = struct.pack("<BB", second, 0)
            packet_id = first
        packet = struct.pack(timestamp_format, timestamp, packet_id) + payload
        body += packet.ljust(packet_size, b"\x00")[:packet_size]
    Path(path).write_bytes(basic + extended + body)


def ramp(number_of_samples, number_of_channels, start=0):
    values = np.arange(start, start + number_of_samples, dtype=np.int64)
    return np.stack([(values + 1000 * index) % 30000 for index in range(number_of_channels)], axis=1)


def in_memory_nwbfile(interfaces):
    converter = ConverterPipe(data_interfaces=interfaces)
    return converter.create_nwbfile(metadata=converter.get_metadata())


@pytest.fixture
def paused(tmp_path):
    """Two blocks 5 s apart on the clock (a pause), a digital event in each, and a spike in each."""
    write_nsx(
        tmp_path / "paused.ns2",
        channels=[channel(1, "chan1"), channel(129, "ainp1", ANALOG)],
        blocks=[(0, ramp(100, 2)), (150_000, ramp(50, 2, start=100))],
    )
    write_nev(
        tmp_path / "paused.nev",
        packets=[digital(30, 7), spike(60, 1), digital(165_000, 8), spike(166_500, 1)],
        electrodes=((1, 250), (129, 152)),
    )
    return tmp_path / "paused.ns2"


@pytest.fixture
def reset(tmp_path):
    """A clock reset: 100 samples from 0, then 50 from 2 ms after the clock restarts."""
    write_nsx(
        tmp_path / "reset.ns2",
        channels=[channel(1, "chan1")],
        blocks=[(0, ramp(100, 1)), (60, ramp(50, 1, start=100))],
    )
    write_nev(tmp_path / "reset.nev", packets=[digital(30, 1), spike(900, 1), digital(90, 2), spike(120, 1)])
    return tmp_path / "reset.ns2"


class TestTimeline:
    def test_session_start_time_is_the_nsx_time_origin_in_utc(self, paused):
        metadata = BlackrockRecordingInterface(file_path=paused).get_metadata()
        assert metadata["NWBFile"]["session_start_time"] == TIME_ORIGIN_DATETIME
        assert metadata["NWBFile"]["session_start_time"].tzinfo is timezone.utc

    def test_pause_is_one_series_per_run_with_a_runs_table(self, paused):
        interface = BlackrockRecordingInterface(file_path=paused)
        nwbfile = interface.create_nwbfile()
        series = [nwbfile.acquisition[name] for name in sorted(nwbfile.acquisition)]
        assert [s.starting_time for s in series] == [0.0, 5.0]
        assert [s.data.shape[0] for s in series] == [100, 50]
        runs = nwbfile.intervals["runs"].to_dataframe()
        assert runs["start_time"].tolist() == [0.0, 5.0]
        assert runs["stop_time"].tolist() == pytest.approx([0.1, 5.05])
        assert runs["after_clock_reset"].tolist() == [False, False]
        assert runs["source_file"].tolist() == ["paused.ns2", "paused.ns2"]

    def test_tables_follow_each_segment_alignment(self, paused):
        interface = BlackrockRecordingInterface(file_path=paused)
        interface.set_aligned_segment_starting_times([10.0, 100.0])
        nwbfile = interface.create_nwbfile()
        # Each segment is moved by its own amount: the second, at 5 s in the file, to 105 s.
        assert list(nwbfile.intervals["runs"]["start_time"][:]) == [10.0, 105.0]
        # 1 ms into the first run and 0.5 s into the second, each moved with its own run.
        assert list(nwbfile.intervals["nev_events"]["start_time"][:]) == pytest.approx([10.001, 105.5])

    def test_contiguous_blocks_are_one_run(self, tmp_path):
        write_nsx(
            tmp_path / "joined.ns2",
            channels=[channel(1, "chan1")],
            blocks=[(0, ramp(100, 1)), (100 * 30, ramp(40, 1, start=100))],
        )
        recording = BlackrockRecordingInterface(file_path=tmp_path / "joined.ns2").recording_extractor
        assert recording.get_num_segments() == 1
        np.testing.assert_array_equal(recording.get_traces(return_in_uV=False)[:, 0], np.arange(140))
        np.testing.assert_array_equal(recording.get_traces(start_frame=95, end_frame=105)[:, 0], np.arange(95, 105))

    def test_clock_reset_is_placed_after_the_run_before(self, reset):
        with pytest.warns(UserWarning, match="was reset 1 time"):
            interface = BlackrockRecordingInterface(file_path=reset)
        recording = interface.recording_extractor
        assert recording.get_start_time(segment_index=1) == pytest.approx(0.1 + 0.002)
        nwbfile = interface.create_nwbfile()
        runs = nwbfile.intervals["runs"].to_dataframe()
        assert runs["after_clock_reset"].tolist() == [False, True]
        events = nwbfile.intervals["nev_events"].to_dataframe()
        # 1 ms after the start, and 3 ms after the reset, which is placed at the end of the first run.
        assert events["start_time"].tolist() == pytest.approx([0.001, 0.103])
        assert events["value"].tolist() == [1, 2]

    def test_clock_that_does_not_start_at_the_recording_is_rebased(self, tmp_path):
        # FileSpec 3.0 clocks keep counting from when the processor started: 893.7 s in the GIN file.
        start = 26_810_899
        write_nsx(
            tmp_path / "late.ns5", channels=[channel(1, "chan1")], blocks=[(start, ramp(30, 1))], period=1, spec="3.0"
        )
        write_nev(tmp_path / "late.nev", packets=[spike(start + 3, 1), digital(start + 15, 3)], spec="3.0")
        interface = BlackrockRecordingInterface(file_path=tmp_path / "late.ns5")
        assert interface.recording_extractor.get_start_time(segment_index=0) == 0.0
        events = interface.create_nwbfile().intervals["nev_events"].to_dataframe()
        assert events["start_time"].tolist() == pytest.approx([15 / 30_000])

        sorting = BlackrockSortingInterface(file_path=tmp_path / "late.nev").sorting_extractor
        assert sorting.get_unit_spike_train(0, return_times=True).tolist() == pytest.approx([3 / 30_000])


class TestFileSetOrigin:
    """Every stream of a set, and its .nev, is placed from the set's earliest first sample."""

    @pytest.fixture
    def two_streams(self, tmp_path):
        # The .ns2's first sample is 2 ms after the .ns5's, as the GIN .ns6 starts 0.134 ms after its .ns4.
        write_nsx(tmp_path / "set.ns5", channels=[channel(1, "chan1")], blocks=[(900, ramp(300, 1))], period=1)
        write_nsx(tmp_path / "set.ns2", channels=[channel(1, "chan1")], blocks=[(960, ramp(10, 1))])
        write_nev(tmp_path / "set.nev", packets=[spike(990, 1), digital(1020, 5)])
        return tmp_path

    def test_streams_share_the_earliest_first_sample(self, two_streams):
        fast = BlackrockRecordingInterface(file_path=two_streams / "set.ns5")
        slow = BlackrockRecordingInterface(file_path=two_streams / "set.ns2")
        assert fast.recording_extractor.get_start_time(segment_index=0) == 0.0
        assert slow.recording_extractor.get_start_time(segment_index=0) == pytest.approx(0.002)
        for interface in (fast, slow):
            events = interface.create_nwbfile().intervals["nev_events"].to_dataframe()
            assert events["start_time"].tolist() == pytest.approx([0.004])

    def test_sorting_gives_the_same_times_from_any_stream(self, two_streams):
        times = [
            BlackrockSortingInterface(file_path=two_streams / "set.nev", nsx_to_load=nsx_to_load)
            .sorting_extractor.get_unit_spike_train(0, return_times=True)
            .tolist()
            for nsx_to_load in (None, 2, 5)
        ]
        assert times == [pytest.approx([0.003])] * 3

    def test_ptp_origin_reads_only_the_first_packets(self, tmp_path, monkeypatch):
        # The earliest of the first _ORIGIN_WINDOW packets: a packet stored late within them counts, one after not.
        ticks = np.r_[5_000_000, np.arange(1, 8) * 1_000_000, 0, np.arange(9, 12) * 1_000_000]
        write_ptp_nsx(tmp_path / "late.ns2", channels=[channel(1, "chan1")], timestamps=ticks, samples=ramp(12, 1))
        assert _blackrock_file_set.first_sample_tick(tmp_path / "late.ns2") == (0, 1e9)
        monkeypatch.setattr(_blackrock_file_set, "_ORIGIN_WINDOW", 8)
        assert _blackrock_file_set.first_sample_tick(tmp_path / "late.ns2") == (1_000_000, 1e9)

    def test_blocks_neo_drops_do_not_set_the_origin(self, tmp_path):
        # neo drops blocks shorter than two samples, so a lone sample before the data does not move t = 0.
        write_nsx(tmp_path / "lone.ns2", channels=[channel(1, "chan1")], blocks=[(0, ramp(1, 1)), (300, ramp(10, 1))])
        assert _blackrock_file_set.first_sample_tick(tmp_path / "lone.ns2") == (300, 30_000.0)
        recording = BlackrockRecordingInterface(file_path=tmp_path / "lone.ns2").recording_extractor
        assert recording.get_start_time(segment_index=0) == 0.0


class TestPTP:
    @pytest.fixture(autouse=True, params=["one_chunk", "chunks_of_4"])
    def chunk_size(self, request, monkeypatch):
        # Long PTP files are scanned and searched a chunk of timestamps at a time; small chunks test the seams.
        if request.param == "chunks_of_4":
            monkeypatch.setattr(_blackrock_file_set, "_CHUNK", 4)

    def write(self, path, timestamps, number_of_channels=1):
        samples = ramp(len(timestamps), number_of_channels)
        write_ptp_nsx(
            path,
            channels=[channel(i + 1, f"chan{i + 1}") for i in range(number_of_channels)],
            timestamps=timestamps,
            samples=samples,
        )
        return samples

    def test_starts_at_zero_not_at_the_unix_epoch(self, tmp_path):
        epoch = 1_688_252_801_327_128_600
        self.write(tmp_path / "ptp.ns2", epoch + np.arange(20) * 1_000_000)
        recording = BlackrockRecordingInterface(file_path=tmp_path / "ptp.ns2").recording_extractor
        assert recording.get_num_segments() == 1
        assert recording.get_start_time(segment_index=0) == 0.0

    def test_out_of_order_packets_are_read_in_timestamp_order(self, tmp_path):
        # As in the GIN "missing samples" files: a block of packets stored after the block that follows it.
        order = np.r_[0:10, 15:20, 10:15, 20:30]
        samples = self.write(tmp_path / "shuffled.ns2", (np.arange(30) * 1_000_000)[order], number_of_channels=2)
        with pytest.warns(UserWarning, match="10 of the packets .* out of timestamp order"):
            recording = BlackrockRecordingInterface(file_path=tmp_path / "shuffled.ns2").recording_extractor
        assert recording.get_num_segments() == 1
        expected = samples[np.argsort(order)]
        np.testing.assert_array_equal(recording.get_traces(return_in_uV=False), expected)
        np.testing.assert_array_equal(
            recording.get_traces(start_frame=8, end_frame=17, channel_ids=["2"])[:, 0], expected[8:17, 1]
        )

    def test_missing_samples_start_a_new_run(self, tmp_path):
        ticks = np.r_[np.arange(10), np.arange(13, 20)] * 1_000_000 + 5
        self.write(tmp_path / "gap.ns2", ticks)
        write_nev(tmp_path / "gap.nev", packets=[digital(int(ticks[12]), 4)], spec="3.0", resolution=1_000_000_000)
        interface = BlackrockRecordingInterface(file_path=tmp_path / "gap.ns2")
        recording = interface.recording_extractor
        assert [recording.get_num_samples(segment_index=i) for i in range(2)] == [10, 7]
        assert recording.get_start_time(segment_index=1) == pytest.approx(0.013)
        events = interface.create_nwbfile().intervals["nev_events"].to_dataframe()
        assert events["start_time"].tolist() == pytest.approx([0.015])

    def test_events_stay_on_their_samples_when_the_clock_drifts(self, tmp_path):
        # 0.1% slow: by sample 1999 the timestamps are 2 ms past the nominal 1 kHz placement.
        ticks = np.arange(2000) * 1_001_000
        self.write(tmp_path / "drift.ns2", ticks)
        write_nev(tmp_path / "drift.nev", packets=[digital(int(ticks[1500]), 9)], spec="3.0", resolution=1_000_000_000)
        with pytest.warns(UserWarning, match="drift up to 1.999 ms"):
            interface = BlackrockRecordingInterface(file_path=tmp_path / "drift.ns2")
        events = interface.create_nwbfile().intervals["nev_events"].to_dataframe()
        assert events["start_time"].tolist() == pytest.approx([1.5])


class TestFileSpec21:
    def write(self, tmp_path, with_nev=True):
        samples = ramp(41, 2)
        write_v21_nsx(tmp_path / "old.ns2", electrode_ids=[1, 129], samples=samples)
        if with_nev:
            write_nev(
                tmp_path / "old.nev",
                packets=[digital(300, 5)],
                electrodes=((1, 250), (129, 21516)),
                spec="2.1",
                application="File Dialog v4.15.0.0",
            )
        return samples

    def test_scaled_with_the_nev_digitization_factors(self, tmp_path):
        self.write(tmp_path)
        recording = BlackrockRecordingInterface(file_path=tmp_path / "old.ns2").recording_extractor
        # 21516 is the overflowed factor of buggy Cerebus firmware, which NPMK replaces.
        np.testing.assert_allclose(recording.get_channel_gains(), [0.25, 152.592547])
        assert recording.get_property("physical_unit").tolist() == ["uV", "uV"]

    def test_every_sample_is_read(self, tmp_path):
        samples = self.write(tmp_path)
        recording = BlackrockRecordingInterface(file_path=tmp_path / "old.ns2").recording_extractor
        np.testing.assert_array_equal(recording.get_traces(return_in_uV=False), samples)

    def test_start_time_comes_from_the_nev_with_a_warning(self, tmp_path):
        self.write(tmp_path)
        interface = BlackrockRecordingInterface(file_path=tmp_path / "old.ns2")
        with pytest.warns(UserWarning, match="File Dialog versions before 6.05"):
            metadata = interface.get_metadata()
        assert metadata["NWBFile"]["session_start_time"] == TIME_ORIGIN_DATETIME

    def test_without_nev_is_refused(self, tmp_path):
        self.write(tmp_path, with_nev=False)
        with pytest.raises(ValueError, match="FileSpec 2.1 file, whose header gives no scaling"):
            BlackrockRecordingInterface(file_path=tmp_path / "old.ns2")


class TestChannels:
    def test_header_gains_and_mixed_units(self, paused):
        recording = BlackrockRecordingInterface(file_path=paused).recording_extractor
        # (max_analog - min_analog) / (max_digital - min_digital), and the analog input's mV taken to µV.
        np.testing.assert_allclose(recording.get_channel_gains(), [16382 / 65528, 10000 / 65534 * 1000])
        np.testing.assert_allclose(recording.get_channel_offsets(), [0.0, 0.0])
        assert recording.get_property("physical_unit").tolist() == ["uV", "mV"]
        np.testing.assert_allclose(recording.get_property("gain_to_physical_unit"), [0.25, 10000 / 65534])

    def test_repeated_labels_are_renamed(self, tmp_path):
        write_nsx(
            tmp_path / "twice.ns3",
            channels=[channel(1, "chan1"), channel(2, "chan1"), channel(3, "chan3")],
            blocks=[(0, ramp(10, 3))],
        )
        with pytest.warns(UserWarning, match="'chan1' -> 'chan1_2'"):
            interface = BlackrockRecordingInterface(file_path=tmp_path / "twice.ns3")
        nwbfile = interface.create_nwbfile()
        assert list(nwbfile.electrodes["channel_name"][:]) == ["chan1", "chan1_2", "chan3"]
        assert list(nwbfile.electrodes["original_channel_name"][:]) == ["chan1", "chan1", "chan3"]


class TestFileNames:
    def test_dotted_base_name(self, tmp_path):
        write_nsx(tmp_path / "rec.2024.01.ns3", channels=[channel(1, "chan1")], blocks=[(0, ramp(10, 1))])
        write_nev(tmp_path / "rec.2024.01.nev", packets=[digital(30, 1)])
        interface = BlackrockRecordingInterface(file_path=tmp_path / "rec.2024.01.ns3")
        assert interface.blackrock_nev.path.name == "rec.2024.01.nev"

    def test_base_name_with_one_nsx(self, tmp_path):
        write_nsx(tmp_path / "rec.2024.ns3", channels=[channel(1, "chan1")], blocks=[(0, ramp(10, 1))])
        interface = BlackrockRecordingInterface(file_path=tmp_path / "rec.2024")
        assert interface.blackrock_stream.path.name == "rec.2024.ns3"

    def test_base_name_with_two_nsx_is_refused(self, tmp_path):
        for number in (2, 5):
            write_nsx(tmp_path / f"rec.ns{number}", channels=[channel(1, "chan1")], blocks=[(0, ramp(10, 1))])
        with pytest.raises(ValueError, match="rec.ns2, rec.ns5"):
            BlackrockRecordingInterface(file_path=tmp_path / "rec")

    def test_nsx_override(self, tmp_path):
        write_nsx(tmp_path / "samples.ns2", channels=[channel(1, "chan1")], blocks=[(0, ramp(10, 1))])
        write_nev(tmp_path / "events.nev", packets=[digital(30, 1)])
        interface = BlackrockRecordingInterface(
            file_path=tmp_path / "events.nev", nsx_override=tmp_path / "samples.ns2"
        )
        assert interface.blackrock_stream.path.name == "samples.ns2"
        assert interface.blackrock_nev.path.name == "events.nev"
        assert interface.recording_extractor.get_num_samples() == 10

    def test_nev_file_path(self, tmp_path):
        write_nsx(tmp_path / "samples.ns2", channels=[channel(1, "chan1")], blocks=[(0, ramp(10, 1))])
        write_nev(tmp_path / "other.nev", packets=[digital(30, 1)])
        interface = BlackrockRecordingInterface(
            file_path=tmp_path / "samples.ns2", nev_file_path=tmp_path / "other.nev"
        )
        assert len(interface.create_nwbfile().intervals["nev_events"]) == 1

    def test_not_an_nsx_file_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="not a Blackrock .ns1 to .ns6 file"):
            BlackrockRecordingInterface(file_path=tmp_path / "rec.edf")


class TestNevEvents:
    def test_digital_serial_and_comments(self, tmp_path):
        write_nsx(tmp_path / "ev.ns2", channels=[channel(1, "chan1")], blocks=[(0, ramp(100, 1))])
        write_nev(
            tmp_path / "ev.nev",
            # In time order, as a .nev is: neo reads a step back in time as a clock reset.
            packets=[
                digital(30, 65535),
                digital(60, 42, reason=129),
                comment(900, b"Comment2\x03\xda\x0b"),  # FileSpec2.3001.nev: the packet's timestamp follows the text
                comment(1200, "Größe".encode("utf-16-le"), char_set=1),
                comment(1500, b"roi:1:enter", char_set=255),
            ],
        )
        events = BlackrockRecordingInterface(file_path=tmp_path / "ev.ns2").create_nwbfile().intervals["nev_events"]
        frame = events.to_dataframe()
        assert frame["start_time"].tolist() == pytest.approx([0.001, 0.002, 0.03, 0.04, 0.05])
        assert frame["stop_time"].tolist() == pytest.approx([0.002, 0.003, 0.031, 0.041, 0.051])
        assert frame["kind"].tolist() == ["digital_input", "serial_input", "comment", "comment", "tracking_event"]
        assert frame["label"].tolist() == ["65535", "42", "Comment2", "Größe", "roi:1:enter"]
        assert frame["value"].tolist() == [65535, 42, -1, -1, -1]
        assert frame["source_file"].tolist() == ["ev.nev"] * 5

    def test_not_written_when_asked(self, paused):
        interface = BlackrockRecordingInterface(file_path=paused)
        nwbfile = mock_NWBFile()
        interface.add_to_nwbfile(nwbfile, write_nev_events=False, write_runs=False)
        assert len(nwbfile.intervals or {}) == 0

    def test_two_recordings_share_the_table_and_one_file_is_written_once(self, paused, reset):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            first = BlackrockRecordingInterface(file_path=paused, metadata_key="a")
            again = BlackrockRecordingInterface(file_path=paused, metadata_key="b")
            other = BlackrockRecordingInterface(file_path=reset, metadata_key="c")
            nwbfile = mock_NWBFile()
            for name, interface in (("A", first), ("B", again), ("C", other)):
                metadata = interface.get_metadata()
                metadata["Ecephys"]["ElectricalSeries"][interface.metadata_key]["name"] = name
                interface.add_to_nwbfile(nwbfile, metadata=metadata)
        sources = nwbfile.intervals["nev_events"].to_dataframe()["source_file"].tolist()
        assert sources == ["paused.nev", "paused.nev", "reset.nev", "reset.nev"]


class TestSorting:
    def test_spikes_are_on_the_recording_timeline(self, reset):
        with pytest.warns(UserWarning, match="was reset"):
            sorting = BlackrockSortingInterface(file_path=reset.with_suffix(".nev")).sorting_extractor
        # 30 ms from the start; 4 ms after the reset, placed at the end of the first run (0.1 s + 2 ms + 2 ms).
        assert sorting.get_unit_spike_train(0, return_times=True).tolist() == pytest.approx([0.03, 0.104])
        assert sorting.get_property("electrode_id").tolist() == [1]
        assert sorting.get_property("unit_number").tolist() == [0]

    def test_session_start_time_prefers_the_nsx(self, paused):
        write_nev(
            paused.with_suffix(".nev"),
            packets=[spike(60, 1)],
            application="File Dialog v6.04.00.00",
            time_origin=(2020, 1, 3, 1, 21, 4, 5, 6),
        )
        metadata = BlackrockSortingInterface(file_path=paused.with_suffix(".nev")).get_metadata()
        assert metadata["NWBFile"]["session_start_time"] == TIME_ORIGIN_DATETIME

    def test_without_nsx_uses_the_nev_clock(self, paused):
        interface = BlackrockSortingInterface(file_path=paused.with_suffix(".nev"), nsx_to_load=[])
        times = interface.sorting_extractor.get_unit_spike_train(0, return_times=True)
        assert times.tolist() == pytest.approx([0.002, 5.55])

    def test_both_interfaces_share_one_parse_and_each_warns(self, reset):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            recording = BlackrockRecordingInterface(file_path=reset)
            sorting = BlackrockSortingInterface(file_path=reset.with_suffix(".nev"))
            # A later interface over the unchanged file warns too, while the first still holds the parsed stream.
            again = BlackrockRecordingInterface(file_path=reset)
        assert sum("was reset" in str(warning.message) for warning in caught) == 3
        assert sorting.blackrock_stream is recording.blackrock_stream is again.blackrock_stream

    def test_recording_and_sorting_in_one_file(self, paused, tmp_path):
        recording = BlackrockRecordingInterface(file_path=paused)
        sorting = BlackrockSortingInterface(file_path=paused.with_suffix(".nev"), nsx_to_load=2)
        nwbfile_path = tmp_path / "both.nwb"
        converter = ConverterPipe(data_interfaces=dict(recording=recording, sorting=sorting))
        converter.run_conversion(nwbfile_path=nwbfile_path, metadata=converter.get_metadata())
        with NWBHDF5IO(nwbfile_path, "r") as io:
            nwbfile = io.read()
            assert nwbfile.session_start_time == TIME_ORIGIN_DATETIME
            assert nwbfile.units["spike_times"][0].tolist() == pytest.approx([0.002, 5.55])
            assert nwbfile.acquisition["ElectricalSeries1"].starting_time == 5.0
            assert list(nwbfile.intervals["nev_events"]["start_time"][:]) == pytest.approx([0.001, 5.5])


class TestBlackrockRecordingInterfaceMixin(RecordingExtractorInterfaceTestMixin):
    data_interface_cls = BlackrockRecordingInterface

    @pytest.fixture
    def setup_interface(self, request, tmp_path):
        write_nsx(
            tmp_path / "mixin.ns3", channels=[channel(1, "chan1"), channel(2, "chan2")], blocks=[(0, ramp(200, 2))]
        )
        self.interface_kwargs = dict(file_path=str(tmp_path / "mixin.ns3"))
        self.interface = self.data_interface_cls(**self.interface_kwargs)
        self.test_name = "synthetic"
        return self.interface, self.test_name

    def check_extracted_metadata(self, metadata: dict):
        assert metadata["NWBFile"]["session_start_time"] == TIME_ORIGIN_DATETIME
        assert metadata["Ecephys"]["ElectricalSeries"] == {"blackrock_recording": dict(name="ElectricalSeries")}
