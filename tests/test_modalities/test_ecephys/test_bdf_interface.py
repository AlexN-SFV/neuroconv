"""Tests for the BDFRecordingInterface.

Builds a small valid BioSemi BDF (24-bit) with pyedflib so the tests run without external data.
"""

import warnings
from datetime import datetime, timezone

import numpy as np
import pytest
from pynwb import NWBHDF5IO

from neuroconv.datainterfaces import BDFRecordingInterface

SAMPLING_FREQUENCY = 256.0
DURATION_SECONDS = 2
CHANNEL_LABELS = ["Fz", "Cz", "Pz", "Status"]
# BDF is 24-bit signed.
BDF_DIGITAL_MIN = -(2**23)
BDF_DIGITAL_MAX = 2**23 - 1
PHYSICAL_MIN = -1000.0
PHYSICAL_MAX = 1000.0


@pytest.fixture
def bdf_path(tmp_path):
    """Write a small valid BDF+ file (byte0 = 0xFF 'BIOSEMI', 24-bit samples)."""
    pyedflib = pytest.importorskip("pyedflib")
    path = tmp_path / "recording.bdf"

    number_of_channels = len(CHANNEL_LABELS)
    number_of_samples = int(SAMPLING_FREQUENCY * DURATION_SECONDS)
    rng = np.random.default_rng(seed=0)
    # Physical values in microvolts, kept within the physical range.
    signals = [
        (rng.standard_normal(number_of_samples) * 100.0).clip(PHYSICAL_MIN, PHYSICAL_MAX)
        for _ in range(number_of_channels)
    ]

    writer = pyedflib.EdfWriter(str(path), number_of_channels, file_type=pyedflib.FILETYPE_BDFPLUS)
    writer.setSignalHeaders(
        [
            dict(
                label=label,
                dimension="uV",
                sample_frequency=SAMPLING_FREQUENCY,
                physical_min=PHYSICAL_MIN,
                physical_max=PHYSICAL_MAX,
                digital_min=BDF_DIGITAL_MIN,
                digital_max=BDF_DIGITAL_MAX,
                transducer="",
                prefilter="",
            )
            for label in CHANNEL_LABELS
        ]
    )
    writer.setPatientCode("subject42")
    writer.setTechnician("tech")
    writer.setStartdatetime(datetime(2023, 5, 1, 9, 30, 0))
    writer.writeSamples([signal.astype(np.float64) for signal in signals])
    writer.close()
    return path, signals


def _write_bdf_with_dimension(path, dimension):
    """Write a one-channel BDF whose physical range of ±1000 is declared in ``dimension``."""
    pyedflib = pytest.importorskip("pyedflib")
    writer = pyedflib.EdfWriter(str(path), 1, file_type=pyedflib.FILETYPE_BDFPLUS)
    writer.setSignalHeaders(
        [
            dict(
                label="Fz",
                dimension=dimension,
                sample_frequency=SAMPLING_FREQUENCY,
                physical_min=PHYSICAL_MIN,
                physical_max=PHYSICAL_MAX,
                digital_min=BDF_DIGITAL_MIN,
                digital_max=BDF_DIGITAL_MAX,
                transducer="",
                prefilter="",
            )
        ]
    )
    writer.setStartdatetime(datetime(2023, 5, 1, 9, 30, 0))
    writer.writeSamples([np.zeros(int(SAMPLING_FREQUENCY * DURATION_SECONDS))])
    writer.close()
    return path


class TestBDFRecordingInterface:
    def test_identity(self):
        assert BDFRecordingInterface.associated_suffixes == (".bdf",)
        assert BDFRecordingInterface.display_name == "BDF Recording"
        assert "BioSemi Data Format" in BDFRecordingInterface.keywords

    def test_is_a_valid_bdf_file(self, bdf_path):
        path, _ = bdf_path
        with open(path, "rb") as file:
            header = file.read(8)
        assert header[0] == 255 and header[1:8] == b"BIOSEMI"

    def test_reads_channels_and_data(self, bdf_path, tmp_path):
        path, signals = bdf_path
        interface = BDFRecordingInterface(file_path=path)

        recording = interface.recording_extractor
        assert list(recording.get_channel_ids()) == CHANNEL_LABELS
        assert recording.get_sampling_frequency() == SAMPLING_FREQUENCY

        metadata = interface.get_metadata()
        metadata["NWBFile"]["session_start_time"] = datetime(2023, 5, 1, 9, 30, tzinfo=timezone.utc)
        nwbfile_path = tmp_path / "out.nwb"
        interface.run_conversion(nwbfile_path=nwbfile_path, metadata=metadata, overwrite=True)

        with NWBHDF5IO(path=str(nwbfile_path), mode="r") as io:
            nwbfile = io.read()
            electrical_series = nwbfile.acquisition["ElectricalSeries"]
            assert electrical_series.data.shape == (int(SAMPLING_FREQUENCY * DURATION_SECONDS), len(CHANNEL_LABELS))
            # Data written in volts; source physical values were microvolts. Tolerance covers 24-bit quantization.
            written_microvolts = electrical_series.data[:] * electrical_series.conversion * 1e6
            np.testing.assert_allclose(written_microvolts, np.array(signals).T, atol=1e-2)

    def test_channels_to_skip(self, bdf_path):
        path, _ = bdf_path
        interface = BDFRecordingInterface(file_path=path, channels_to_skip=["Status"])
        assert list(interface.recording_extractor.get_channel_ids()) == ["Fz", "Cz", "Pz"]

    def test_header_metadata(self, bdf_path):
        # session_start_time, experimenter, and subject_id are pulled from the BDF header.
        path, _ = bdf_path
        interface = BDFRecordingInterface(file_path=path)
        metadata = interface.get_metadata()
        assert metadata["NWBFile"]["session_start_time"] == datetime(2023, 5, 1, 9, 30, 0)
        assert metadata["NWBFile"]["experimenter"] == ["tech"]
        assert metadata["Subject"]["subject_id"] == "subject42"

    def test_get_available_channel_ids(self, bdf_path):
        path, _ = bdf_path
        assert BDFRecordingInterface.get_available_channel_ids(path) == CHANNEL_LABELS

    # EDF and BDF headers are ASCII, so the micro sign cannot appear in one; its readings are tested on the
    # shared map through the BrainVision tests.
    @pytest.mark.parametrize(
        "dimension, to_microvolts",
        [("uV", 1.0), ("nV", 1e-3), ("mV", 1e3), ("mvolt", 1e3), ("V", 1e6), ("degC", 1.0)],
    )
    def test_physical_dimension_scales_the_gain_to_microvolts(self, tmp_path, dimension, to_microvolts):
        path = _write_bdf_with_dimension(tmp_path / "recording.bdf", dimension)
        gain = BDFRecordingInterface(file_path=path).recording_extractor.get_channel_gains()[0]

        # The gain in the declared unit is the physical range over the digital range.
        gain_in_dimension = (PHYSICAL_MAX - PHYSICAL_MIN) / (BDF_DIGITAL_MAX - BDF_DIGITAL_MIN)
        assert gain == pytest.approx(gain_in_dimension * to_microvolts)


class TestRepeatedLabels:
    @pytest.fixture
    def repeated_path(self, tmp_path):
        """Two channels labelled EXG1, as a BioSemi montage that lists one input twice can leave behind."""
        pyedflib = pytest.importorskip("pyedflib")
        path = tmp_path / "repeated.bdf"
        labels = ["EXG1", "EXG1", "Cz"]
        number_of_samples = int(SAMPLING_FREQUENCY * DURATION_SECONDS)
        signals = [np.full(number_of_samples, 10.0 * (index + 1)) for index in range(len(labels))]
        writer = pyedflib.EdfWriter(str(path), len(labels), file_type=pyedflib.FILETYPE_BDFPLUS)
        writer.setSignalHeaders(
            [
                dict(
                    label=label,
                    dimension="uV",
                    sample_frequency=SAMPLING_FREQUENCY,
                    physical_min=PHYSICAL_MIN,
                    physical_max=PHYSICAL_MAX,
                    digital_min=BDF_DIGITAL_MIN,
                    digital_max=BDF_DIGITAL_MAX,
                    transducer="",
                    prefilter="",
                )
                for label in labels
            ]
        )
        writer.setStartdatetime(datetime(2023, 5, 1, 9, 30, 0))
        writer.writeSamples(signals)
        writer.close()
        return path

    def test_a_repeat_is_renamed_and_written(self, repeated_path, tmp_path):
        assert BDFRecordingInterface.get_available_channel_ids(repeated_path) == ["EXG1", "EXG1_2", "Cz"]
        with pytest.warns(UserWarning, match="'EXG1' -> 'EXG1_2'"):
            interface = BDFRecordingInterface(file_path=repeated_path)
        recording = interface.recording_extractor
        assert list(recording.get_channel_ids()) == ["EXG1", "EXG1_2", "Cz"]
        # The second EXG1 reads its own signal (20 uV), not the first's.
        assert recording.get_traces(channel_ids=["EXG1_2"], return_in_uV=True)[0, 0] == pytest.approx(20.0, abs=1e-3)

        metadata = interface.get_metadata()
        metadata["NWBFile"]["session_start_time"] = datetime(2023, 5, 1, 9, 30, tzinfo=timezone.utc)
        nwbfile_path = tmp_path / "repeated.nwb"
        interface.run_conversion(nwbfile_path=nwbfile_path, metadata=metadata, overwrite=True)
        with NWBHDF5IO(nwbfile_path, "r") as io:
            electrodes = io.read().electrodes.to_dataframe()
        assert list(electrodes["channel_name"]) == ["EXG1", "EXG1_2", "Cz"]
        assert list(electrodes["original_channel_name"]) == ["EXG1", "EXG1", "Cz"]

    def test_skipping_by_label_or_by_renamed_id(self, repeated_path):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            by_id = BDFRecordingInterface(file_path=repeated_path, channels_to_skip=["EXG1_2"])
            by_label = BDFRecordingInterface(file_path=repeated_path, channels_to_skip=["EXG1"])
        assert list(by_id.recording_extractor.get_channel_ids()) == ["EXG1", "Cz"]
        assert list(by_label.recording_extractor.get_channel_ids()) == ["Cz"]

    def test_rename_warning_names_the_callers_line(self, repeated_path):
        from pathlib import Path

        with pytest.warns(UserWarning, match="renamed") as caught:
            BDFRecordingInterface(file_path=repeated_path)
        assert [Path(warning.filename) for warning in caught if "renamed" in str(warning.message)] == [Path(__file__)]
