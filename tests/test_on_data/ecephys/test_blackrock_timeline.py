"""Where the Blackrock GIN files land on the NWB timeline, and what their .nev files hold.

The values were checked against the files' headers and, for the clock and time zone, against what the files say
of themselves: ``20231027-125608-001`` was named at 12:56:08 EDT, its ``TimeOrigin`` 16:56:08 UTC.
"""

import warnings
from datetime import datetime, timezone

import numpy as np
import pytest
from pynwb.testing.mock.file import mock_NWBFile

from neuroconv.datainterfaces import (
    BlackrockRecordingInterface,
    BlackrockSortingInterface,
)

try:
    from ..setup_paths import ECEPHY_DATA_PATH
except ImportError:
    from setup_paths import ECEPHY_DATA_PATH

BLACKROCK = ECEPHY_DATA_PATH / "blackrock"


def recording_interface(relative_path, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return BlackrockRecordingInterface(file_path=BLACKROCK / relative_path, **kwargs)


def test_session_start_time_is_utc():
    metadata = recording_interface("blackrock_3_0_ptp/20231027-125608-001.ns2").get_metadata()
    assert metadata["NWBFile"]["session_start_time"] == datetime(2023, 10, 27, 16, 56, 8, 173000, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    "relative_path, clock_start",
    [
        ("blackrock_3_0/file_spec_3_0.ns6", 893.6966333),
        ("blackrock_3_0_ptp/20231027-125608-001.ns2", 1688252801.327129),
    ],
)
def test_first_sample_is_at_zero(relative_path, clock_start):
    # The clock value neo and SpikeInterface give the first sample, which used to be written as its starting time.
    interface = recording_interface(relative_path)
    assert interface.blackrock_stream.runs[0].first_tick / interface.blackrock_stream.timeline.resolution == (
        pytest.approx(clock_start)
    )
    assert interface.recording_extractor.get_start_time(segment_index=0) == 0.0


def test_ptp_packets_out_of_order_are_reordered_not_split():
    path = BLACKROCK / "blackrock_ptp_with_missing_samples" / "Hub1-NWBtestfile_neural_wspikes.ns6"
    with pytest.warns(UserWarning, match="58 of the packets .* out of timestamp order"):
        interface = BlackrockRecordingInterface(file_path=path)
    recording = interface.recording_extractor
    # neo, with gap_tolerance_ms=0.5, splits it into 632, 58 and 310 samples, the middle segment holding a step
    # back in time. Sorted by timestamp no sample is missing: 1000 samples over 33.3 ms at 30 kHz.
    assert recording.get_num_segments() == 1
    assert recording.get_num_samples() == 1000


def test_streams_of_one_set_share_the_earliest_first_sample():
    # The .ns4 and .ns6 share their TimeOrigin to the millisecond, but the .ns6 starts 133542 ns later.
    folder = "blackrock_ptp_with_missing_samples"
    first, second = (
        recording_interface(f"{folder}/Hub1-NWBtestfile_neural_wspikes.{suffix}") for suffix in ("ns4", "ns6")
    )
    assert first.blackrock_stream.time_origin == second.blackrock_stream.time_origin
    assert first.recording_extractor.get_start_time(segment_index=0) == 0.0
    assert second.recording_extractor.get_start_time(segment_index=0) == pytest.approx(133542e-9)


def test_pause_and_reset():
    paused = recording_interface("segment/PauseCorrect/pause_correct.ns2").recording_extractor
    assert [paused.get_start_time(segment_index=i) for i in range(2)] == pytest.approx([0.0, 31.0087])
    with pytest.warns(UserWarning, match="reset 1 time"):
        reset = BlackrockRecordingInterface(file_path=BLACKROCK / "segment" / "ResetCorrect" / "reset.ns2")
    # 4020 samples at 1 kHz, then the clock restarts and the next block is stamped 96 ticks (3.2 ms).
    assert [reset.recording_extractor.get_start_time(segment_index=i) for i in range(2)] == pytest.approx(
        [0.0, 4.02 + 0.0032]
    )


def test_file_spec_21_uses_the_nev_and_reads_every_sample():
    recording = recording_interface("blackrock_2_1/l101210-001.ns2").recording_extractor
    assert recording.get_num_samples() == 3641  # neo gives 3640, one short of the file size
    np.testing.assert_allclose(recording.get_channel_gains(), 152.592547)
    assert set(recording.get_property("physical_unit")) == {"uV"}


def test_nev_events_and_comments():
    interface = recording_interface("FileSpec2.3001.ns5")
    nwbfile = mock_NWBFile()
    interface.add_to_nwbfile(nwbfile)
    events = nwbfile.intervals["nev_events"].to_dataframe()
    assert (events["kind"] == "digital_input").sum() == 190
    comments = events[events["kind"] == "comment"]
    assert comments["label"].tolist() == ["Comment", "Comment2", "Comment3"]
    assert comments["start_time"].tolist() == pytest.approx([355816 / 30000, 776707 / 30000, 856210 / 30000])


def test_sorting_spikes_on_the_recording_timeline():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sorting = BlackrockSortingInterface(file_path=BLACKROCK / "blackrock_3_0_ptp" / "20231027-125608-001.nev")
    times = np.concatenate(
        [sorting.sorting_extractor.get_unit_spike_train(unit, return_times=True) for unit in sorting.units_ids]
    )
    # 1688252801.47 s on the PTP clock, 0.144 s after the .ns2's first sample.
    assert times.min() == pytest.approx(0.14406676, abs=1 / 30000)
    assert times.max() < 2.149
