"""Tests for the BrainVisionRecordingInterface and its reader.

Each test writes a small BrainVision recording with :func:`write_brainvision`, so no external data is needed.
Where a test reproduces a real exporter's quirk, the header text follows that exporter's output verbatim.
"""

from datetime import datetime

import numpy as np
import pytest
from pynwb import NWBHDF5IO

from neuroconv.datainterfaces import BrainVisionRecordingInterface
from neuroconv.datainterfaces.ecephys.brainvision._brainvision_extractor import (
    BrainVisionRecordingExtractor,
)
from neuroconv.datainterfaces.ecephys.brainvision._brainvision_reader import (
    read_brainvision_header,
)

SAMPLING_FREQUENCY = 1000.0
NUMBER_OF_SAMPLES = 300


def _channel_line(index, name, resolution="1", unit="µV", reference=""):
    fields = [name, reference, resolution]
    if unit is not None:
        fields.append(unit)
    return f"Ch{index}=" + ",".join(fields)


def write_brainvision(
    folder,
    data,
    *,
    stem="recording",
    binary_format="INT_16",
    orientation="MULTIPLEXED",
    channels=None,
    markers=None,
    codepage="UTF-8",
    encoding="utf-8",
    sampling_interval=1e6 / SAMPLING_FREQUENCY,
    data_file=None,
    marker_file=None,
    extra_common="",
    bom=False,
    header_first_line="Brain Vision Data Exchange Header File Version 1.0",
    data_points=None,
):
    """
    Write ``<stem>.vhdr``, ``<stem>.vmrk`` and ``<stem>.eeg`` and return the header's path.

    ``data`` is ``(samples, channels)``. ``channels`` is a list of ``Ch`` line bodies (without the ``ChN=``
    prefix); by default each channel is named ``C<n>`` at resolution 1 in microvolts. ``markers`` is a list of
    ``Mk`` line bodies. A ``codepage`` of None omits the line.
    """
    dtype = {"INT_16": "<i2", "INT_32": "<i4", "IEEE_FLOAT_32": "<f4"}[binary_format]
    number_of_channels = data.shape[1]
    if channels is None:
        channels = [f"C{index + 1},,1,µV" for index in range(number_of_channels)]
    data_file = data_file or f"{stem}.eeg"
    marker_file = marker_file or f"{stem}.vmrk"

    stored = np.asarray(data, dtype=dtype)
    if orientation == "VECTORIZED":
        stored = stored.T
    (folder / f"{stem}.eeg").write_bytes(np.ascontiguousarray(stored).tobytes())

    codepage_line = f"Codepage={codepage}\n" if codepage is not None else ""
    data_points_line = f"DataPoints={data_points}\n" if data_points is not None else ""
    header = (
        f"{header_first_line}\n"
        "; written by the neuroconv test suite\n\n"
        "[Common Infos]\n"
        f"{codepage_line}"
        f"DataFile={data_file}\n"
        f"MarkerFile={marker_file}\n"
        "DataFormat=BINARY\n"
        f"DataOrientation={orientation}\n"
        f"NumberOfChannels={number_of_channels}\n"
        f"{data_points_line}"
        f"SamplingInterval={sampling_interval}\n"
        f"{extra_common}"
        "\n[Binary Infos]\n"
        f"BinaryFormat={binary_format}\n\n"
        "[Channel Infos]\n"
        "; Each entry: Ch<Channel number>=<Name>,<Reference channel name>,\n"
        + "".join(f"Ch{index + 1}={body}\n" for index, body in enumerate(channels))
        + "\n[Comment]\nFree text = with ; anything in it\n"
    )
    marker_text = (
        "Brain Vision Data Exchange Marker File, Version 1.0\n\n"
        "[Common Infos]\n"
        f"{codepage_line}"
        f"DataFile={data_file}\n\n"
        "[Marker Infos]\n"
        "; Each entry: Mk<Marker number>=<Type>,<Description>,<Position in data points>,\n"
        + "".join(f"Mk{index + 1}={body}\n" for index, body in enumerate(markers or []))
    )
    prefix = b"\xef\xbb\xbf" if bom else b""
    (folder / f"{stem}.vhdr").write_bytes(prefix + header.encode(encoding))
    (folder / f"{stem}.vmrk").write_bytes(prefix + marker_text.encode(encoding))
    return folder / f"{stem}.vhdr"


@pytest.fixture
def int16_data():
    rng = np.random.default_rng(seed=0)
    return rng.integers(-2000, 2000, size=(NUMBER_OF_SAMPLES, 3)).astype("int16")


class TestReadingSamples:
    @pytest.mark.parametrize("binary_format", ["INT_16", "INT_32", "IEEE_FLOAT_32"])
    @pytest.mark.parametrize("orientation", ["MULTIPLEXED", "VECTORIZED"])
    def test_traces_match_the_stored_samples(self, tmp_path, int16_data, binary_format, orientation):
        path = write_brainvision(tmp_path, int16_data, binary_format=binary_format, orientation=orientation)
        recording = BrainVisionRecordingExtractor(path)

        assert recording.get_num_samples() == NUMBER_OF_SAMPLES
        assert recording.get_sampling_frequency() == SAMPLING_FREQUENCY
        np.testing.assert_array_equal(recording.get_traces(), int16_data)
        np.testing.assert_array_equal(
            recording.get_traces(start_frame=17, end_frame=40, channel_ids=["C3", "C1"]), int16_data[17:40, [2, 0]]
        )

    def test_resolution_applies_to_floating_point_samples(self, tmp_path, int16_data):
        """neo forces the gain of float data to 1; a pybv file at resolution 0.1 must still scale by 0.1."""
        channels = [f"C{index + 1},,0.1,µV" for index in range(3)]
        path = write_brainvision(tmp_path, int16_data, binary_format="IEEE_FLOAT_32", channels=channels)
        recording = BrainVisionRecordingExtractor(path)

        np.testing.assert_allclose(recording.get_channel_gains(), [0.1, 0.1, 0.1])
        np.testing.assert_allclose(recording.get_traces(return_in_uV=True), int16_data * 0.1, rtol=1e-6)

    @pytest.mark.parametrize(
        "unit, gain",
        [("µV", 0.5), ("μV", 0.5), ("uV", 0.5), ("nV", 0.5e-3), ("mV", 0.5e3), ("V", 0.5e6)],
    )
    def test_unit_scales_the_gain_to_microvolts(self, tmp_path, int16_data, unit, gain):
        channels = [f"C{index + 1},,0.5,{unit}" for index in range(3)]
        recording = BrainVisionRecordingExtractor(write_brainvision(tmp_path, int16_data, channels=channels))

        np.testing.assert_allclose(recording.get_channel_gains(), [gain] * 3)
        assert list(recording.get_property("physical_unit")) == [unit] * 3
        np.testing.assert_allclose(recording.get_property("gain_to_physical_unit"), [0.5] * 3)

    def test_missing_unit_means_microvolts(self, tmp_path, int16_data):
        channels = ["C1,,2", "C2,,2,", "C3,,2,µV"]
        recording = BrainVisionRecordingExtractor(write_brainvision(tmp_path, int16_data, channels=channels))

        np.testing.assert_allclose(recording.get_channel_gains(), [2.0, 2.0, 2.0])
        assert list(recording.get_property("physical_unit")) == ["µV"] * 3

    def test_mixed_units_in_one_file(self, tmp_path, int16_data):
        """As in NEMAR nm000397: EEG channels at 0.25 µV beside a dynamometer channel in mV."""
        channels = ["C1,,0.25,µV", "C2,,0.25,µV", "dynamometer,,0.15259254737998595,mV"]
        recording = BrainVisionRecordingExtractor(write_brainvision(tmp_path, int16_data, channels=channels))

        np.testing.assert_allclose(recording.get_channel_gains(), [0.25, 0.25, 152.59254737998595])
        assert not np.any(recording.get_channel_offsets())

    def test_blank_resolution_is_taken_as_one_and_warns(self, tmp_path, int16_data):
        channels = ["C1,,,µV", "C2,,1,µV", "C3,,1,µV"]
        with pytest.warns(UserWarning, match="no resolution"):
            recording = BrainVisionRecordingExtractor(write_brainvision(tmp_path, int16_data, channels=channels))
        np.testing.assert_allclose(recording.get_channel_gains(), [1.0, 1.0, 1.0])

    def test_non_voltage_unit_warns(self, tmp_path, int16_data):
        channels = ["C1,,1,µV", "C2,,1,µV", "Temp,,0.1,C"]
        with pytest.warns(UserWarning, match="not a voltage"):
            BrainVisionRecordingExtractor(write_brainvision(tmp_path, int16_data, channels=channels))

    def test_escaped_commas_in_channel_names(self, tmp_path, int16_data):
        channels = [r"A\1B,,1,µV", "C2,Ref\\1X,1,µV", "C3,,1,µV"]
        recording = BrainVisionRecordingExtractor(write_brainvision(tmp_path, int16_data, channels=channels))

        assert list(recording.get_channel_ids()) == ["A,B", "C2", "C3"]
        assert list(recording.get_property("reference")) == ["", "Ref,X", ""]

    def test_channels_to_skip(self, tmp_path, int16_data):
        path = write_brainvision(tmp_path, int16_data)
        recording = BrainVisionRecordingExtractor(path, channels_to_skip=["C2"])

        assert list(recording.get_channel_ids()) == ["C1", "C3"]
        np.testing.assert_array_equal(recording.get_traces(), int16_data[:, [0, 2]])
        with pytest.raises(ValueError, match="not in the recording"):
            BrainVisionRecordingExtractor(path, channels_to_skip=["nope"])

    def test_recording_pickles(self, tmp_path, int16_data):
        import pickle

        recording = BrainVisionRecordingExtractor(write_brainvision(tmp_path, int16_data))
        restored = pickle.loads(pickle.dumps(recording))
        np.testing.assert_array_equal(restored.get_traces(), int16_data)


class TestHeaderText:
    def test_latin1_header_without_codepage(self, tmp_path, int16_data):
        """AnyWave's exporter (OpenNeuro ds005169, ds006519) writes 'µ' as the single byte 0xB5 and no Codepage."""
        markers = ["Comment,B'03-B'04 0.75mA; Simple visual hallucinations,33,1,0"]
        path = write_brainvision(
            tmp_path,
            int16_data,
            channels=["P'01-P'02,,1,µV", "H'02-H'03,,1,µV", "K'01-K'02,,1,µV"],
            markers=markers,
            codepage=None,
            encoding="latin-1",
            header_first_line="Brain Vision Data Exchange Header File Version 1.0",
        )
        header = read_brainvision_header(path)

        assert header.units == ["µV"] * 3
        assert header.channel_names[0] == "P'01-P'02"
        assert header.markers[0].description == "B'03-B'04 0.75mA; Simple visual hallucinations"

    def test_ansi_codepage_is_windows_1252(self, tmp_path, int16_data):
        path = write_brainvision(tmp_path, int16_data, codepage="ANSI", encoding="cp1252")
        assert read_brainvision_header(path).units == ["µV"] * 3

    def test_utf8_byte_order_mark(self, tmp_path, int16_data):
        path = write_brainvision(tmp_path, int16_data, bom=True)
        header = read_brainvision_header(path)
        assert header.number_of_channels == 3
        assert header.units == ["µV"] * 3

    def test_sampling_interval_is_in_microseconds(self, tmp_path, int16_data):
        path = write_brainvision(tmp_path, int16_data, sampling_interval=488.28125)
        assert read_brainvision_header(path).sampling_frequency == 2048.0

    @pytest.mark.parametrize(
        "extra_common, error, match",
        [
            ("DataType=FREQUENCYDOMAIN\n", NotImplementedError, "TIMEDOMAIN"),
            ("SegmentationType=MARKERBASED\n", NotImplementedError, "segmented"),
        ],
    )
    def test_unsupported_layouts_are_refused(self, tmp_path, int16_data, extra_common, error, match):
        path = write_brainvision(tmp_path, int16_data, extra_common=extra_common)
        with pytest.raises(error, match=match):
            read_brainvision_header(path)

    def test_ascii_data_is_refused(self, tmp_path, int16_data):
        path = write_brainvision(tmp_path, int16_data)
        path.write_text(path.read_text(encoding="utf-8").replace("DataFormat=BINARY", "DataFormat=ASCII"), "utf-8")
        with pytest.raises(NotImplementedError, match="ASCII"):
            read_brainvision_header(path)

    def test_passing_the_data_file_points_at_the_header(self, tmp_path, int16_data):
        write_brainvision(tmp_path, int16_data)
        with pytest.raises(ValueError, match=r"recording\.vhdr"):
            read_brainvision_header(tmp_path / "recording.eeg")

    def test_renamed_companions_are_found_by_base_name(self, tmp_path, int16_data):
        """BIDS conversion renames the files without updating the references inside them."""
        path = write_brainvision(tmp_path, int16_data, data_file="old.eeg", marker_file="old.vmrk")
        with pytest.warns(UserWarning, match="does not exist"):
            header = read_brainvision_header(path)
        assert header.data_path.name == "recording.eeg"
        assert header.marker_path.name == "recording.vmrk"

    def test_missing_data_file_raises(self, tmp_path, int16_data):
        path = write_brainvision(tmp_path, int16_data)
        (tmp_path / "recording.eeg").unlink()
        with pytest.raises(FileNotFoundError):
            read_brainvision_header(path)

    def test_truncated_multiplexed_file_reads_the_complete_samples(self, tmp_path, int16_data):
        path = write_brainvision(tmp_path, int16_data)
        data_path = tmp_path / "recording.eeg"
        # Three channels of int16 make 6-byte samples, so cutting 3 bytes leaves the last sample incomplete.
        data_path.write_bytes(data_path.read_bytes()[:-3])
        with pytest.warns(UserWarning, match="ends partway"):
            recording = BrainVisionRecordingExtractor(path)
        np.testing.assert_array_equal(recording.get_traces(), int16_data[:-1])

    def test_truncated_vectorized_file_raises(self, tmp_path, int16_data):
        path = write_brainvision(tmp_path, int16_data, orientation="VECTORIZED", data_points=NUMBER_OF_SAMPLES)
        data_path = tmp_path / "recording.eeg"
        data_path.write_bytes(data_path.read_bytes()[:-6])
        with pytest.raises(ValueError, match="truncated"):
            read_brainvision_header(path)


class TestMarkers:
    def test_positions_count_from_one(self, tmp_path, int16_data):
        path = write_brainvision(tmp_path, int16_data, markers=["Stimulus,S  1,1,1,0", "Stimulus,S  2,101,5,2"])
        markers = read_brainvision_header(path).markers

        assert [marker.sample_index for marker in markers] == [0, 100]
        assert markers[1].size == 5
        assert markers[1].channel_number == 2

    def test_descriptions_keep_equals_signs_and_escaped_commas(self, tmp_path, int16_data):
        markers = ["Comment,a=b,11,1,0", r"Comment,left\1right,21,1,0"]
        header = read_brainvision_header(write_brainvision(tmp_path, int16_data, markers=markers))
        assert [marker.description for marker in header.markers] == ["a=b", "left,right"]

    def test_unescaped_comma_in_a_description(self, tmp_path, int16_data):
        header = read_brainvision_header(write_brainvision(tmp_path, int16_data, markers=["Comment,one, two,11,1,0"]))
        marker = header.markers[0]
        assert (marker.description, marker.sample_index, marker.size) == ("one, two", 10, 1)

    def test_marker_without_size_or_channel(self, tmp_path, int16_data):
        header = read_brainvision_header(write_brainvision(tmp_path, int16_data, markers=["Stimulus,S  1,11"]))
        marker = header.markers[0]
        assert (marker.sample_index, marker.size, marker.channel_number) == (10, 0, 0)

    def test_unreadable_marker_is_skipped_with_a_warning(self, tmp_path, int16_data):
        path = write_brainvision(tmp_path, int16_data, markers=["Stimulus,S  1,eleven,1,0", "Stimulus,S  2,21,1,0"])
        with pytest.warns(UserWarning, match="unreadable marker"):
            header = read_brainvision_header(path)
        assert [marker.description for marker in header.markers] == ["S  2"]


class TestSegments:
    def test_new_segment_date_is_the_recording_start(self, tmp_path, int16_data):
        path = write_brainvision(tmp_path, int16_data, markers=["New Segment,,1,1,0,20210602161337327145"])
        header = read_brainvision_header(path)

        assert len(header.segments) == 1
        assert header.recording_start == datetime(2021, 6, 2, 16, 13, 37, 327145)

    @pytest.mark.parametrize("date", ["19000101000000000000", ""])
    def test_placeholder_or_absent_date_is_not_a_recording_start(self, tmp_path, int16_data, date):
        """Wonambi (OpenNeuro ds003688) writes 1900-01-01 into anonymised recordings."""
        marker = "New Segment,,1,1,0" + (f",{date}" if date else "")
        header = read_brainvision_header(write_brainvision(tmp_path, int16_data, markers=[marker]))
        assert header.recording_start is None

    def test_paused_recording_is_split_at_its_real_times(self, tmp_path, int16_data):
        markers = [
            "New Segment,,1,1,0,20210101120000000000",
            "Stimulus,S  1,151,1,0",
            "New Segment,,101,1,0,20210101120100000000",  # resumed one minute after the first started
        ]
        recording = BrainVisionRecordingExtractor(write_brainvision(tmp_path, int16_data, markers=markers))

        assert recording.get_num_segments() == 2
        assert recording.get_num_samples(segment_index=0) == 100
        assert recording.get_num_samples(segment_index=1) == 200
        assert recording.get_start_time(segment_index=0) == 0.0
        assert recording.get_start_time(segment_index=1) == 60.0
        np.testing.assert_array_equal(recording.get_traces(segment_index=1), int16_data[100:])

    def test_undated_segments_are_placed_back_to_back(self, tmp_path, int16_data):
        markers = ["New Segment,,1,1,0", "New Segment,,101,1,0"]
        recording = BrainVisionRecordingExtractor(write_brainvision(tmp_path, int16_data, markers=markers))
        assert recording.get_start_time(segment_index=1) == pytest.approx(0.1)

    def test_segment_dated_before_the_previous_one_ends_warns(self, tmp_path, int16_data):
        markers = ["New Segment,,1,1,0,20210101120000000000", "New Segment,,101,1,0,20210101120000050000"]
        path = write_brainvision(tmp_path, int16_data, markers=markers)
        with pytest.warns(UserWarning, match="earlier than the end"):
            recording = BrainVisionRecordingExtractor(path)
        assert recording.get_start_time(segment_index=1) == pytest.approx(0.1)


class TestBrainVisionRecordingInterface:
    def test_metadata(self, tmp_path, int16_data):
        path = write_brainvision(tmp_path, int16_data, markers=["New Segment,,1,1,0,20210602161337000000"])
        interface = BrainVisionRecordingInterface(file_path=path)
        metadata = interface.get_metadata()

        assert metadata["NWBFile"]["session_start_time"] == datetime(2021, 6, 2, 16, 13, 37)
        assert "brainvision_recording" in metadata["Ecephys"]["ElectricalSeries"]
        interface.validate_metadata(metadata)

    def test_get_available_channel_ids(self, tmp_path, int16_data):
        path = write_brainvision(tmp_path, int16_data)
        assert BrainVisionRecordingInterface.get_available_channel_ids(path) == ["C1", "C2", "C3"]

    def test_conversion_writes_data_and_markers(self, tmp_path, int16_data):
        channels = ["C1,,0.5,µV", "C2,,0.5,µV", "C3,,0.5,µV"]
        markers = [
            "New Segment,,1,1,0,20210602161337000000",
            "Stimulus,S  1,11,1,0",
            "Comment,note,101,50,2",
        ]
        path = write_brainvision(tmp_path, int16_data, channels=channels, markers=markers)
        interface = BrainVisionRecordingInterface(file_path=path)
        nwbfile_path = tmp_path / "out.nwb"
        interface.run_conversion(nwbfile_path=nwbfile_path, metadata=interface.get_metadata())

        with NWBHDF5IO(nwbfile_path, "r") as io:
            nwbfile = io.read()
            series = nwbfile.acquisition["ElectricalSeries"]
            np.testing.assert_array_equal(series.data[:], int16_data)
            assert series.conversion == pytest.approx(0.5e-6)
            assert series.rate == SAMPLING_FREQUENCY

            table = nwbfile.intervals["markers"].to_dataframe()
            assert list(table["marker_type"]) == ["Stimulus", "Comment"]
            assert list(table["label"]) == ["S  1", "note"]
            np.testing.assert_allclose(table["start_time"], [0.010, 0.100])
            np.testing.assert_allclose(table["stop_time"], [0.011, 0.150])
            assert list(table["size_in_samples"]) == [1, 50]
            assert list(table["channel"]) == ["", "C2"]
            assert "segments" not in nwbfile.intervals

    def test_paused_recording_writes_one_series_per_segment(self, tmp_path, int16_data):
        markers = [
            "New Segment,,1,1,0,20210101120000000000",
            "New Segment,,101,1,0,20210101120100000000",
            "Stimulus,S  1,151,1,0",
        ]
        path = write_brainvision(tmp_path, int16_data, markers=markers)
        interface = BrainVisionRecordingInterface(file_path=path)
        nwbfile_path = tmp_path / "out.nwb"
        interface.run_conversion(nwbfile_path=nwbfile_path, metadata=interface.get_metadata())

        with NWBHDF5IO(nwbfile_path, "r") as io:
            nwbfile = io.read()
            series = [nwbfile.acquisition[name] for name in sorted(nwbfile.acquisition)]
            assert [s.starting_time for s in series] == [0.0, 60.0]
            np.testing.assert_array_equal(series[1].data[:], int16_data[100:])

            # Marker at sample 150 sits 50 samples into the second segment.
            np.testing.assert_allclose(nwbfile.intervals["markers"].to_dataframe()["start_time"], [60.05])
            segments = nwbfile.intervals["segments"].to_dataframe()
            np.testing.assert_allclose(segments["start_time"], [0.0, 60.0])
            np.testing.assert_allclose(segments["stop_time"], [0.1, 60.2])

    def test_markers_follow_alignment(self, tmp_path, int16_data):
        path = write_brainvision(tmp_path, int16_data, markers=["Stimulus,S  1,11,1,0"])
        interface = BrainVisionRecordingInterface(file_path=path)
        interface.set_aligned_starting_time(aligned_starting_time=5.0)
        metadata = interface.get_metadata()
        metadata["NWBFile"]["session_start_time"] = datetime(2021, 1, 1)
        nwbfile_path = tmp_path / "out.nwb"
        interface.run_conversion(nwbfile_path=nwbfile_path, metadata=metadata)

        with NWBHDF5IO(nwbfile_path, "r") as io:
            np.testing.assert_allclose(io.read().intervals["markers"].to_dataframe()["start_time"], [5.01])

    def test_zero_size_marker_still_spans_its_sample(self, tmp_path, int16_data):
        """Wonambi (OpenNeuro ds003688) writes size 0; an interval must still end after it starts."""
        path = write_brainvision(tmp_path, int16_data, markers=["Stimulus,start task,11,0,0"])
        interface = BrainVisionRecordingInterface(file_path=path)
        metadata = interface.get_metadata()
        metadata["NWBFile"]["session_start_time"] = datetime(2021, 1, 1)
        nwbfile_path = tmp_path / "out.nwb"
        interface.run_conversion(nwbfile_path=nwbfile_path, metadata=metadata)

        with NWBHDF5IO(nwbfile_path, "r") as io:
            table = io.read().intervals["markers"].to_dataframe()
            np.testing.assert_allclose(table["stop_time"] - table["start_time"], [1 / SAMPLING_FREQUENCY])
            assert list(table["size_in_samples"]) == [0]

    def test_markers_can_be_left_out(self, tmp_path, int16_data):
        path = write_brainvision(tmp_path, int16_data, markers=["Stimulus,S  1,11,1,0"])
        interface = BrainVisionRecordingInterface(file_path=path)
        metadata = interface.get_metadata()
        metadata["NWBFile"]["session_start_time"] = datetime(2021, 1, 1)
        nwbfile_path = tmp_path / "out.nwb"
        interface.run_conversion(nwbfile_path=nwbfile_path, metadata=metadata, write_markers=False)

        with NWBHDF5IO(nwbfile_path, "r") as io:
            nwbfile = io.read()
            assert nwbfile.intervals is None or "markers" not in nwbfile.intervals
