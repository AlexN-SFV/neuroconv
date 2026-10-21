"""Tests for the BrainVisionRecordingInterface and its reader.

Each test writes a small BrainVision recording with :func:`write_brainvision`, so no external data is needed.
Where a test reproduces a real exporter's quirk, the header text follows that exporter's output verbatim.
"""

import warnings
from datetime import datetime

import numpy as np
import pytest
from pynwb import NWBHDF5IO
from pynwb.epoch import TimeIntervals
from pynwb.testing.mock.file import mock_NWBFile

from neuroconv import ConverterPipe
from neuroconv.datainterfaces import BrainVisionRecordingInterface
from neuroconv.datainterfaces.ecephys.brainvision._brainvision_extractor import (
    BrainVisionRecordingExtractor,
)
from neuroconv.datainterfaces.ecephys.brainvision._brainvision_reader import (
    read_brainvision_header,
)
from neuroconv.tools.testing.data_interface_mixins import (
    RecordingExtractorInterfaceTestMixin,
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

    def test_blank_resolution_on_floating_point_data_is_silent(self, tmp_path, int16_data):
        """Analyzer leaves the resolution blank on every float export, whose samples are already in the unit."""
        channels = ["C1,,,µV", "C2,,,µV", "C3,,,µV"]
        path = write_brainvision(tmp_path, int16_data, binary_format="IEEE_FLOAT_32", channels=channels)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            recording = BrainVisionRecordingExtractor(path)
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

    def test_utf8_text_declared_ansi_is_read_as_utf8(self, tmp_path, int16_data):
        """Writers declare ANSI over UTF-8 text; decoding it as Windows-1252 garbles every non-ASCII character."""
        path = write_brainvision(
            tmp_path,
            int16_data,
            channels=["Ö1,,1,µV", "C2,,1,µV", "C3,,1,µV"],
            markers=["Comment,Übung,11,1,0"],
            codepage="ANSI",
            encoding="utf-8",
        )
        header = read_brainvision_header(path)

        assert header.channel_names[0] == "Ö1"
        assert header.units == ["µV"] * 3
        assert header.markers[0].description == "Übung"

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
            ("SegmentationType=FIXTIME\n", NotImplementedError, "SegmentationType=FIXTIME"),
            ("Averaged=YES\n", NotImplementedError, "average over epochs"),
        ],
    )
    def test_unsupported_layouts_are_refused(self, tmp_path, int16_data, extra_common, error, match):
        path = write_brainvision(tmp_path, int16_data, extra_common=extra_common)
        with pytest.raises(error, match=match):
            read_brainvision_header(path)

    @pytest.mark.parametrize(
        "old, new, match",
        [
            ("NumberOfChannels=3", "NumberOfChannels=0", "at least one channel"),
            ("NumberOfChannels=3", "NumberOfChannels=three", "NumberOfChannels as 'three', which is not an integer"),
            ("SamplingInterval=1000.0", "SamplingInterval=fast", "SamplingInterval as 'fast', which is not a number"),
            ("SamplingInterval=1000.0", "SamplingInterval=0", "must be positive"),
            ("Ch1=C1,,1,µV", "Ch1=C1,,one,µV", "resolution of Ch1 as 'one', which is not a number"),
        ],
    )
    def test_malformed_numbers_name_the_file_and_field(self, tmp_path, int16_data, old, new, match):
        path = write_brainvision(tmp_path, int16_data)
        path.write_text(path.read_text(encoding="utf-8").replace(old, new), encoding="utf-8")
        with pytest.raises(ValueError, match=f"recording.vhdr.*{match}"):
            read_brainvision_header(path)

    def test_decimal_comma_resolution_warns(self, tmp_path, int16_data):
        """'0,1' meaning 0.1 splits into a resolution of 0 and a unit of '1'."""
        path = write_brainvision(tmp_path, int16_data, channels=["C1,,0,1,µV", "C2,,1,µV", "C3,,1,µV"])
        with pytest.warns(UserWarning, match="resolution of 0"):
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

    def test_later_segments_are_placed_from_the_dated_segment_before_them(self, tmp_path, int16_data):
        markers = [
            "New Segment,,1,1,0",
            "New Segment,,101,1,0,20210101120000000000",
            "New Segment,,201,1,0,20210101120030000000",  # resumed 30 s after the second segment started
        ]
        path = write_brainvision(tmp_path, int16_data, markers=markers)
        with pytest.warns(UserWarning, match="nothing to measure"):
            header = read_brainvision_header(path)

        # The first dated segment has no earlier date to be placed by, so the pause before it is unknown.
        assert header.recording_start is None
        assert [segment.start_time for segment in header.segments] == pytest.approx([0.0, 0.1, 30.1])

    def test_segment_dated_before_the_previous_one_ends_warns(self, tmp_path, int16_data):
        markers = ["New Segment,,1,1,0,20210101120000000000", "New Segment,,101,1,0,20210101120000050000"]
        path = write_brainvision(tmp_path, int16_data, markers=markers)
        with pytest.warns(UserWarning, match="earlier than the end"):
            recording = BrainVisionRecordingExtractor(path)
        assert recording.get_start_time(segment_index=1) == pytest.approx(0.1)


class TestEpochedExports:
    """BrainVision Analyzer's SegmentationType=MARKERBASED exports, laid out as GIN's File_brainvision_2 and _3."""

    EPOCHED = "SegmentationType=MARKERBASED\nSegmentDataPoints=100\n"

    def test_epochs_are_placed_at_their_dates(self, tmp_path, int16_data):
        markers = [
            "New Segment,,1,1,0,20111027115127145000",
            "Time 0,,41,1,0",
            "New Segment,,101,1,0,20111027115139115000",  # 11.97 s after the first epoch began
            "New Segment,,201,1,0,20111027115149970000",
        ]
        path = write_brainvision(tmp_path, int16_data, markers=markers, extra_common=self.EPOCHED)
        header = read_brainvision_header(path)

        assert header.is_epoched
        assert header.recording_start == datetime(2011, 10, 27, 11, 51, 27, 145000)
        assert [segment.number_of_samples for segment in header.segments] == [100, 100, 100]
        assert [segment.start_time for segment in header.segments] == pytest.approx([0.0, 11.97, 22.825])

    def test_overlapping_epochs_keep_their_dates(self, tmp_path, int16_data):
        """Epochs cut from overlapping windows overlap in time; each is its own series, so that is kept."""
        markers = [
            "New Segment,,1,1,0,20200101120000000000",
            "New Segment,,101,1,0,20200101120000050000",  # 50 ms after the first, which lasts 100 ms
            "New Segment,,201,1,0,20200101120000100000",
        ]
        path = write_brainvision(tmp_path, int16_data, markers=markers, extra_common=self.EPOCHED)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            header = read_brainvision_header(path)
        assert [segment.start_time for segment in header.segments] == pytest.approx([0.0, 0.05, 0.1])

    def test_epoch_dated_before_the_first_is_placed_after_the_previous(self, tmp_path, int16_data):
        markers = [
            "New Segment,,1,1,0,20200101120000000000",
            "New Segment,,101,1,0,20200101115959000000",
            "New Segment,,201,1,0,20200101120001000000",
        ]
        path = write_brainvision(tmp_path, int16_data, markers=markers, extra_common=self.EPOCHED)
        with pytest.warns(UserWarning, match="earlier than the first epoch"):
            header = read_brainvision_header(path)
        assert [segment.start_time for segment in header.segments] == pytest.approx([0.0, 0.1, 1.0])

    @pytest.mark.parametrize(
        "markers, extra_common, match",
        [
            (["New Segment,,1,1,0", "New Segment,,151,1,0"], EPOCHED, "do not open exactly those epochs"),
            (["New Segment,,1,1,0", "New Segment,,101,1,0"], EPOCHED, "do not open exactly those epochs"),
            (["New Segment,,1,1,0"], "SegmentationType=MARKERBASED\nSegmentDataPoints=120\n", "whole number"),
        ],
    )
    def test_markers_that_do_not_match_the_segmentation_are_refused(
        self, tmp_path, int16_data, markers, extra_common, match
    ):
        path = write_brainvision(tmp_path, int16_data, markers=markers, extra_common=extra_common)
        with pytest.raises(NotImplementedError, match=match):
            read_brainvision_header(path)

    def test_missing_segment_length_says_the_line_is_missing(self, tmp_path, int16_data):
        path = write_brainvision(
            tmp_path, int16_data, markers=["New Segment,,1,1,0"], extra_common="SegmentationType=MARKERBASED\n"
        )
        with pytest.raises(ValueError, match="has no SegmentDataPoints in \\[Common Infos\\]"):
            read_brainvision_header(path)

    def test_blank_segment_length_names_the_value(self, tmp_path, int16_data):
        extra_common = "SegmentationType=MARKERBASED\nSegmentDataPoints=\n"
        path = write_brainvision(tmp_path, int16_data, markers=["New Segment,,1,1,0"], extra_common=extra_common)
        with pytest.raises(ValueError, match="gives SegmentDataPoints as '', which is not an integer"):
            read_brainvision_header(path)

    def test_epochs_are_written_as_series_and_listed(self, tmp_path, int16_data):
        markers = [
            "New Segment,,1,1,0,20200101120000000000",
            "Time 0,,21,1,0",
            "New Segment,,101,1,0,20200101120010000000",
            "Time 0,,121,1,0",
            "New Segment,,201,1,0,20200101120020000000",
            "Time 0,,221,1,0",
        ]
        path = write_brainvision(tmp_path, int16_data, markers=markers, extra_common=self.EPOCHED)
        interface = BrainVisionRecordingInterface(file_path=path)
        nwbfile = mock_NWBFile()
        interface.add_to_nwbfile(nwbfile=nwbfile, metadata=interface.get_metadata())

        series = sorted(nwbfile.acquisition.values(), key=lambda series: series.starting_time)
        assert [series.starting_time for series in series] == [0.0, 10.0, 20.0]
        assert [series.data.maxshape[0] for series in series] == [100, 100, 100]
        segments = nwbfile.intervals["segments"]
        assert list(segments["is_epoch"][:]) == [True, True, True]
        np.testing.assert_allclose(nwbfile.intervals["markers"]["start_time"][:], [0.02, 10.02, 20.02])


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

    def test_markers_outside_the_data_are_skipped_with_a_warning(self, tmp_path, int16_data):
        markers = ["Stimulus,zero,0,1,0", "Stimulus,inside,11,1,0", f"Stimulus,past,{NUMBER_OF_SAMPLES + 1},1,0"]
        path = write_brainvision(tmp_path, int16_data, markers=markers)
        interface = BrainVisionRecordingInterface(file_path=path)
        nwbfile = mock_NWBFile()
        with pytest.warns(UserWarning, match="Skipping 2 marker"):
            interface.add_to_nwbfile(nwbfile=nwbfile, metadata=interface.get_metadata())

        assert list(nwbfile.intervals["markers"]["label"][:]) == ["inside"]

    def test_markers_are_written_in_time_order(self, tmp_path, int16_data):
        """Mk numbers follow the order markers were set, not the order they occur in."""
        path = write_brainvision(tmp_path, int16_data, markers=["Comment,later,201,1,0", "Comment,earlier,11,1,0"])
        interface = BrainVisionRecordingInterface(file_path=path)
        nwbfile = mock_NWBFile()
        interface.add_to_nwbfile(nwbfile=nwbfile, metadata=interface.get_metadata())

        assert list(nwbfile.intervals["markers"]["label"][:]) == ["earlier", "later"]

    def test_two_recordings_in_one_converter_keep_both_sets_of_markers(self, tmp_path, int16_data):
        first_folder, second_folder = tmp_path / "first", tmp_path / "second"
        first_folder.mkdir()
        second_folder.mkdir()
        paused = ["New Segment,,1,1,0", "Stimulus,A,11,1,0", "New Segment,,101,1,0"]
        first = write_brainvision(first_folder, int16_data, stem="run1", markers=paused)
        second = write_brainvision(second_folder, int16_data, stem="run2", markers=paused[:1] + ["Stimulus,B,21,1,0"])

        first_interface = BrainVisionRecordingInterface(file_path=first)
        second_interface = BrainVisionRecordingInterface(file_path=second, metadata_key="run2")
        second_interface.set_aligned_starting_time(aligned_starting_time=100.0)
        converter = ConverterPipe(data_interfaces=dict(run1=first_interface, run2=second_interface))
        metadata = converter.get_metadata()
        metadata["NWBFile"]["session_start_time"] = datetime(2021, 1, 1)
        metadata["Ecephys"]["ElectricalSeries"]["run2"]["name"] = "ElectricalSeriesRun2"
        nwbfile_path = tmp_path / "out.nwb"
        converter.run_conversion(nwbfile_path=nwbfile_path, metadata=metadata)

        with NWBHDF5IO(nwbfile_path, "r") as io:
            nwbfile = io.read()
            assert "ElectricalSeriesRun2" in nwbfile.acquisition
            markers = nwbfile.intervals["markers"].to_dataframe()
            assert list(markers["label"]) == ["A", "B"]
            assert list(markers["source_file"]) == ["run1.vhdr", "run2.vhdr"]
            np.testing.assert_allclose(markers["start_time"], [0.01, 100.02])
            # Only the first recording was paused, so only it has segments to list.
            segments = nwbfile.intervals["segments"].to_dataframe()
            assert list(segments["source_file"]) == ["run1.vhdr", "run1.vhdr"]

    def test_a_foreign_markers_table_is_left_alone_with_a_warning(self, tmp_path, int16_data):
        path = write_brainvision(tmp_path, int16_data, markers=["Stimulus,S  1,11,1,0"])
        interface = BrainVisionRecordingInterface(file_path=path)
        nwbfile = mock_NWBFile()
        nwbfile.add_time_intervals(TimeIntervals(name="markers", description="Written by something else."))
        with pytest.warns(UserWarning, match="already has an intervals table named 'markers'"):
            interface.add_to_nwbfile(nwbfile=nwbfile, metadata=interface.get_metadata())

        assert len(nwbfile.intervals["markers"]) == 0

    PAUSED = ["New Segment,,1,1,0", "Stimulus,A,11,1,0", "New Segment,,101,1,0", "Stimulus,B,151,1,0"]

    def test_separately_built_interfaces_over_one_recording_write_its_rows_once(self, tmp_path, int16_data):
        """An EEG interface beside an auxiliary one with the complementary channels share the recording's rows."""
        path = write_brainvision(tmp_path, int16_data, markers=self.PAUSED)
        interfaces = [
            BrainVisionRecordingInterface(file_path=path, metadata_key="eeg", channels_to_skip=["C3"]),
            BrainVisionRecordingInterface(file_path=path, metadata_key="aux", channels_to_skip=["C1", "C2"]),
        ]
        markers, segments = _convert_through_pipe(tmp_path, interfaces)

        assert list(markers["label"]) == ["A", "B"]
        assert list(segments["segment_index"]) == [0, 1]
        assert list(segments["source_file"]) == ["recording.vhdr"] * 2

    def test_same_name_and_markers_in_two_directories_keeps_both_recordings_rows(self, tmp_path, int16_data):
        """BIDS gives two subjects' recordings the same file name, and a fixed protocol the same markers."""
        interfaces = []
        for subject in ("sub01", "sub02"):
            (tmp_path / subject).mkdir()
            path = write_brainvision(tmp_path / subject, int16_data, markers=self.PAUSED)
            interfaces.append(BrainVisionRecordingInterface(file_path=path, metadata_key=subject))
        markers, segments = _convert_through_pipe(tmp_path, interfaces)

        assert list(markers["label"]) == ["A", "B"] * 2
        assert list(segments["segment_index"]) == [0, 1] * 2

    def test_split_by_offset_leaves_a_brainvision_recording_whole(self, tmp_path, int16_data):
        """The format has no offset, so the reader sets every channel's to 0 and there is one partition."""
        interface = BrainVisionRecordingInterface(file_path=write_brainvision(tmp_path, int16_data))
        assert interface.split_by_offset() == [interface]

    def test_split_by_offset_partitions_write_rows_once(self, tmp_path, int16_data):
        """
        Were a recording ever partitioned, its partitions would share its rows.

        The offsets are set by hand, since the format has none to give a recording more than one.
        """
        interface = BrainVisionRecordingInterface(
            file_path=write_brainvision(tmp_path, int16_data, markers=self.PAUSED)
        )
        interface.recording_extractor.set_channel_offsets(offsets=[0.0, 0.0, 5.0])
        sub_interfaces = interface.split_by_offset()
        assert len(sub_interfaces) == 2
        markers, segments = _convert_through_pipe(tmp_path, sub_interfaces)

        assert list(markers["label"]) == ["A", "B"]
        assert list(segments["segment_index"]) == [0, 1]

    def test_conversion_options_reach_the_writer(self, tmp_path, int16_data):
        path = write_brainvision(tmp_path, int16_data)
        interface = BrainVisionRecordingInterface(file_path=path, channels_to_skip=["C2"])
        nwbfile = mock_NWBFile()
        interface.add_to_nwbfile(
            nwbfile=nwbfile, metadata=interface.get_metadata(), stub_test=True, parent_container="processing/LFP"
        )

        series = nwbfile.processing["ecephys"]["LFP"].electrical_series["ElectricalSeries"]
        assert list(nwbfile.electrodes["channel_name"][:]) == ["C1", "C3"]
        assert series.data.shape[1] == 2
        assert series.data.shape[0] < NUMBER_OF_SAMPLES

    def test_deprecated_write_as(self, tmp_path, int16_data):
        path = write_brainvision(tmp_path, int16_data)
        interface = BrainVisionRecordingInterface(file_path=path)
        nwbfile = mock_NWBFile()
        with pytest.warns(FutureWarning):
            interface.add_to_nwbfile(nwbfile=nwbfile, metadata=interface.get_metadata(), write_as="lfp")
        assert "ElectricalSeries" in nwbfile.processing["ecephys"]["LFP"].electrical_series


def _convert_through_pipe(tmp_path, interfaces):
    """Write the interfaces into one NWB file through a ConverterPipe; return its markers and segments tables."""
    converter = ConverterPipe(data_interfaces={interface.metadata_key: interface for interface in interfaces})
    metadata = converter.get_metadata()
    metadata["NWBFile"]["session_start_time"] = datetime(2021, 1, 1)
    for interface in interfaces:
        metadata["Ecephys"]["ElectricalSeries"][interface.metadata_key]["name"] = f"{interface.metadata_key}_series"

    nwbfile_path = tmp_path / "pipe.nwb"
    converter.run_conversion(nwbfile_path=nwbfile_path, metadata=metadata, overwrite=True)
    with NWBHDF5IO(nwbfile_path, "r") as io:
        nwbfile = io.read()
        # Every segment of every interface reaches the file as its own series, whatever happens to the tables.
        assert len(nwbfile.acquisition) == sum(interface.number_of_segments for interface in interfaces)
        return nwbfile.intervals["markers"].to_dataframe(), nwbfile.intervals["segments"].to_dataframe()


class _SyntheticRecordingForMixin:
    """
    Writes the recording a mixin-based class converts, once per class, into pytest's managed temp directory.

    The mixins read their source from ``interface_kwargs`` on the class, so the fixture sets it there.
    """

    markers: list[str]

    @pytest.fixture(scope="class", autouse=True)
    @classmethod
    def _write_recording(cls, tmp_path_factory):
        folder = tmp_path_factory.mktemp(cls.__name__)
        data = np.random.default_rng(seed=1).integers(-2000, 2000, size=(NUMBER_OF_SAMPLES, 4)).astype("int16")
        channels = [f"C{index + 1},,0.5,µV" for index in range(data.shape[1])]
        path = write_brainvision(folder, data, channels=channels, markers=cls.markers)
        cls.interface_kwargs = dict(file_path=str(path))
        cls.save_directory = folder


class TestBrainVisionRecordingInterfaceContract(_SyntheticRecordingForMixin, RecordingExtractorInterfaceTestMixin):
    data_interface_cls = BrainVisionRecordingInterface
    markers = ["New Segment,,1,1,0,20210602161337000000", "Stimulus,S  1,11,1,0"]

    def check_extracted_metadata(self, metadata: dict):
        assert metadata["NWBFile"]["session_start_time"] == datetime(2021, 6, 2, 16, 13, 37)
        assert "brainvision_recording" in metadata["Ecephys"]["ElectricalSeries"]


class TestPausedBrainVisionRecordingInterfaceContract(
    _SyntheticRecordingForMixin, RecordingExtractorInterfaceTestMixin
):
    data_interface_cls = BrainVisionRecordingInterface
    markers = [
        "New Segment,,1,1,0,20210101120000000000",
        "Stimulus,S  1,151,1,0",
        "New Segment,,101,1,0,20210101120100000000",
    ]

    def check_extracted_metadata(self, metadata: dict):
        assert metadata["NWBFile"]["session_start_time"] == datetime(2021, 1, 1, 12, 0, 0)


class TestRepeatedChannelNames:
    CHANNELS = ["Fp1,,1,µV", "Fp1,,1,µV", "Fp1_2,,1,µV"]

    def test_a_repeat_is_renamed_around_an_existing_name(self, tmp_path, int16_data):
        """The second Fp1 cannot become Fp1_2, which a real channel already has, so it becomes Fp1_3."""
        path = write_brainvision(tmp_path, int16_data, channels=self.CHANNELS)
        assert BrainVisionRecordingInterface.get_available_channel_ids(path) == ["Fp1", "Fp1_3", "Fp1_2"]
        with pytest.warns(UserWarning, match="'Fp1' -> 'Fp1_3'"):
            recording = BrainVisionRecordingExtractor(path)

        assert list(recording.get_channel_ids()) == ["Fp1", "Fp1_3", "Fp1_2"]
        assert list(recording.get_property("original_channel_name")) == ["Fp1", "Fp1", "Fp1_2"]
        np.testing.assert_array_equal(recording.get_traces(channel_ids=["Fp1_3"])[:, 0], int16_data[:, 1])

    def test_skipping_by_name_or_by_renamed_id(self, tmp_path, int16_data):
        path = write_brainvision(tmp_path, int16_data, channels=self.CHANNELS)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            by_id = BrainVisionRecordingExtractor(path, channels_to_skip=["Fp1_3"])
            by_name = BrainVisionRecordingExtractor(path, channels_to_skip=["Fp1"])
        assert list(by_id.get_channel_ids()) == ["Fp1", "Fp1_2"]
        assert "original_channel_name" not in by_id.get_property_keys()
        assert list(by_name.get_channel_ids()) == ["Fp1_2"]

    def test_electrodes_table_keeps_both_channels(self, tmp_path, int16_data):
        path = write_brainvision(tmp_path, int16_data, channels=self.CHANNELS)
        with pytest.warns(UserWarning, match="renamed"):
            interface = BrainVisionRecordingInterface(file_path=path)
        metadata = interface.get_metadata()
        metadata["NWBFile"]["session_start_time"] = datetime(2021, 1, 1)
        nwbfile_path = tmp_path / "out.nwb"
        interface.run_conversion(nwbfile_path=nwbfile_path, metadata=metadata)

        with NWBHDF5IO(nwbfile_path, "r") as io:
            electrodes = io.read().electrodes.to_dataframe()
        assert list(electrodes["channel_name"]) == ["Fp1", "Fp1_3", "Fp1_2"]
        assert list(electrodes["original_channel_name"]) == ["Fp1", "Fp1", "Fp1_2"]
