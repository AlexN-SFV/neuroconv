"""
Tests for EDF+ signals that carry floating-point or long-integer data.

The fixtures are written byte by byte, so these run without external data, network access, or a
particular reader being installed beyond what the ``edf`` extra already provides.
"""

import numpy as np
import pytest

from neuroconv.datainterfaces import EDFRecordingInterface
from neuroconv.datainterfaces.ecephys.edf._edf_log_transform import (
    LogTransform,
    _parse_log_transform,
    read_log_transforms,
)

A = 0.002
YMIN = 0.01
PREFILTER = "sign*LN[sign*(uV      )/(0.01    )]/(0.002   )"

# Values spanning a range no 16-bit integer could hold, which is the point of the transformation.
VALUES = [0.0, 0.05, 1.0, 250.0, 1e4, 5e6, -0.05, -1.0, -250.0, -1e4]


def encode(values, a=A, ymin=YMIN):
    """
    Encode physical values as the integer counts a "Filtered" signal stores.

    ``N = round(ln(Y/Ymin)/a)`` above ``Ymin``, 0 within ``+/-Ymin``, and the negative mirror below.
    """
    counts = np.zeros(len(values), dtype="int16")
    for index, value in enumerate(values):
        if value > ymin:
            counts[index] = round(np.log(value / ymin) / a)
        elif value < -ymin:
            counts[index] = -round(np.log(-value / ymin) / a)
    return counts


def _field(value, width):
    text = str(value)
    assert len(text) <= width, f"header field {text!r} does not fit in {width} bytes"
    # latin-1, so a fixture can write the micro sign a real writer may put in a dimension field.
    return text.ljust(width).encode("latin-1")


def write_edf(
    path,
    *,
    values=tuple(VALUES),
    prefilter=PREFILTER,
    dimension="Filtered",
    transformed_unit_channel="logch",
    include_linear_channel=True,
):
    """Write a continuous EDF+ file with one transformed signal, optionally beside a linear one."""
    counts = encode(list(values))
    samples_per_record = len(counts)
    annotation_samples = 32
    labels = [transformed_unit_channel] + (["linch"] if include_linear_channel else [])
    dimensions = [dimension] + (["uV"] if include_linear_channel else [])
    physical_min = [-32767.0] + ([-1000.0] if include_linear_channel else [])
    physical_max = [32767.0] + ([1000.0] if include_linear_channel else [])
    prefilters = [prefilter] + ([""] if include_linear_channel else [])
    # EDF+ requires an annotations signal, and pyedflib refuses a file without one.
    labels.append("EDF Annotations")
    dimensions.append("")
    physical_min.append(-1.0)
    physical_max.append(1.0)
    prefilters.append("")
    samples = [samples_per_record] * (len(labels) - 1) + [annotation_samples]
    # The specification puts the transformed signal's digital range at +/-32767; EDFlib insists the other
    # signals, the annotations one especially, use the full -32768..32767.
    number_of_signals = len(labels)
    header_size = 256 * (number_of_signals + 1)

    header = bytearray()
    header += _field(0, 8) + _field("X X X X", 80) + _field("Startdate 16-SEP-2021 X X test", 80)
    header += _field("16.09.21", 8) + _field("12.35.13", 8) + _field(header_size, 8)
    header += _field("EDF+C", 44) + _field(1, 8) + _field(1.0, 8) + _field(number_of_signals, 4)
    for label in labels:
        header += _field(label, 16)
    for _ in labels:
        header += _field("", 80)  # transducer
    for value in dimensions:
        header += _field(value, 8)
    for value in physical_min:
        header += _field(value, 8)
    for value in physical_max:
        header += _field(value, 8)
    for index in range(number_of_signals):
        header += _field(-32767 if index == 0 else -32768, 8)
    for _ in labels:
        header += _field(32767, 8)
    for value in prefilters:
        header += _field(value, 80)
    for value in samples:
        header += _field(value, 8)
    for _ in labels:
        header += _field("", 32)  # per-signal reserved
    assert len(header) == header_size

    linear = np.arange(samples_per_record, dtype="int16")
    # The mandatory time-keeping TAL for the single data record.
    annotation_block = b"+0.000000\x14\x14\x00".ljust(annotation_samples * 2, b"\x00")
    body = counts.tobytes() + (linear.tobytes() if include_linear_channel else b"") + annotation_block
    path = str(path)
    with open(path, "wb") as file:
        file.write(bytes(header) + body)
    return path, counts, linear


class TestParseLogTransform:
    def test_parameters_are_parsed(self):
        transform = _parse_log_transform(dimension="Filtered", prefilter=PREFILTER)
        assert transform.unit == "uV"
        assert transform.minimum == pytest.approx(YMIN)
        assert transform.coefficient == pytest.approx(A)

    @pytest.mark.parametrize("dimension", ["uV", "V", "mV", "", "filter"])
    def test_ordinary_signals_are_not_transformed(self, dimension):
        assert _parse_log_transform(dimension=dimension, prefilter="") is None

    @pytest.mark.parametrize("dimension", ["Filtered", "filtered", " FILTERED "])
    def test_marker_is_case_and_space_insensitive(self, dimension):
        assert _parse_log_transform(dimension=dimension, prefilter=PREFILTER) is not None

    @pytest.mark.parametrize(
        "prefilter",
        ["", "HP:0.1Hz LP:75Hz", "sign*LN[sign*(uV)/(0)]/(0.002)", "sign*LN[sign*(uV)/(0.01)]/(abc)"],
    )
    def test_unreadable_parameters_are_refused(self, prefilter):
        """
        The samples are logarithms, so without Ymin and a they cannot be interpreted at all.

        Refusing is the only safe answer: applying the header's linear gain instead is precisely what
        reads the file wrong today.
        """
        with pytest.raises(ValueError, match="logarithmically transformed"):
            _parse_log_transform(dimension="Filtered", prefilter=prefilter)

    def test_round_trip_recovers_the_values(self):
        transform = LogTransform(unit="uV", minimum=YMIN, coefficient=A)
        decoded = transform.decode(encode(VALUES))
        # Accuracy is inherently about a/2 in relative terms — the trade the transformation makes.
        np.testing.assert_allclose(decoded, VALUES, rtol=A, atol=1e-12)

    def test_zero_is_exact(self):
        """``N == 0`` means exactly zero, not ``Ymin``."""
        transform = LogTransform(unit="uV", minimum=YMIN, coefficient=A)
        assert transform.decode(np.array([0], dtype="int16"))[0] == 0.0


class TestReadLogTransforms:
    def test_transformed_signal_is_found(self, tmp_path):
        path, _, _ = write_edf(tmp_path / "log.edf")
        assert list(read_log_transforms(file_path=path)) == ["logch"]

    def test_ordinary_file_yields_nothing(self, tmp_path):
        path, _, _ = write_edf(tmp_path / "plain.edf", dimension="uV", prefilter="")
        assert read_log_transforms(file_path=path) == {}


class TestEDFRecordingInterfaceDecoding:
    def test_values_are_decoded(self, tmp_path):
        """
        Without this the header's linear gain is applied to logarithms.

        A signal spanning 0.05 to 5e6 µV reads back as roughly 24 to 306 µV — no error, no warning.
        """
        path, _, linear = write_edf(tmp_path / "log.edf")
        interface = EDFRecordingInterface(file_path=path)
        recording = interface.recording_extractor

        assert list(interface.log_transformed_channels) == ["logch"]
        # float64, not float32: the specification's own accuracy table reaches 1.4e68.
        assert recording.get_dtype() == np.dtype("float64")

        traces = recording.get_traces()
        # A decoded channel's gain is only the unit scaling and its offset zero, since the exponential
        # relationship cannot be expressed as the scalar gain+offset an ElectricalSeries applies.
        assert recording.get_channel_gains()[0] == pytest.approx(1.0)  # uV -> uV
        assert recording.get_channel_offsets()[0] == 0.0
        np.testing.assert_allclose(traces[:, 0], VALUES, rtol=A, atol=1e-12)

        # The linear signal beside it keeps its raw digital values and the parent reader's gain.
        np.testing.assert_array_equal(traces[:, 1], linear)
        # neo divides by the number of digital codes, 32767 - (-32768) + 1.
        assert recording.get_channel_gains()[1] == pytest.approx(2000.0 / 65536)

    def test_ordinary_file_is_not_wrapped(self, tmp_path):
        """The common path must be untouched, including the recording's dtype."""
        path, _, _ = write_edf(tmp_path / "plain.edf", dimension="uV", prefilter="")
        interface = EDFRecordingInterface(file_path=path)

        assert interface.log_transformed_channels == {}
        assert interface.recording_extractor.get_dtype() == np.dtype("int16")
        assert type(interface.recording_extractor).__name__ == "EDFRecordingExtractor"

    def test_chunked_and_subset_reads_decode_consistently(self, tmp_path):
        path, _, _ = write_edf(tmp_path / "log.edf")
        recording = EDFRecordingInterface(file_path=path).recording_extractor
        full = recording.get_traces()

        for start, end in [(0, 1), (2, 5), (4, len(VALUES))]:
            np.testing.assert_array_equal(recording.get_traces(start_frame=start, end_frame=end), full[start:end])
        # The transformation must follow the channel, not its column position.
        np.testing.assert_array_equal(recording.get_traces(channel_ids=["linch", "logch"]), full[:, [1, 0]])
        np.testing.assert_array_equal(recording.get_traces(channel_ids=["logch"]), full[:, [0]])

    def test_original_unit_sets_the_gain(self, tmp_path):
        """The transformation displaces the unit from the dimension field, so it comes from the prefilter."""
        path, _, _ = write_edf(tmp_path / "log.edf", prefilter="sign*LN[sign*(mV      )/(0.01    )]/(0.002   )")
        recording = EDFRecordingInterface(file_path=path).recording_extractor
        assert recording.get_channel_gains()[0] == pytest.approx(1e3)  # mV -> uV

    def test_non_voltage_unit_warns_and_assumes_microvolts(self, tmp_path):
        path, _, _ = write_edf(tmp_path / "log.edf", prefilter="sign*LN[sign*(%       )/(0.01    )]/(0.002   )")
        with pytest.warns(UserWarning, match="not a voltage"):
            recording = EDFRecordingInterface(file_path=path).recording_extractor
        assert recording.get_channel_gains()[0] == pytest.approx(1.0)

    def test_unreadable_parameters_can_be_skipped(self, tmp_path):
        path, _, _ = write_edf(tmp_path / "log.edf", prefilter="HP:0.1Hz")
        with pytest.raises(ValueError, match="logarithmically transformed"):
            EDFRecordingInterface(file_path=path)

        interface = EDFRecordingInterface(file_path=path, channels_to_skip=["logch"])
        assert list(interface.recording_extractor.get_channel_ids()) == ["linch"]

    def test_metadata_still_comes_from_the_header(self, tmp_path):
        """The decoding wrapper does not forward neo_reader, so the header is captured while reading."""
        from datetime import datetime

        path, _, _ = write_edf(tmp_path / "log.edf")
        metadata = EDFRecordingInterface(file_path=path).get_metadata()
        assert metadata["NWBFile"]["session_start_time"] == datetime(2021, 9, 16, 12, 35, 13)

    def test_conversion_writes_decoded_values(self, tmp_path):
        from pynwb import NWBHDF5IO

        path, _, _ = write_edf(tmp_path / "log.edf")
        interface = EDFRecordingInterface(file_path=path)
        metadata = interface.get_metadata()

        nwbfile_path = tmp_path / "log.nwb"
        interface.run_conversion(nwbfile_path=str(nwbfile_path), metadata=metadata, overwrite=True)

        with NWBHDF5IO(str(nwbfile_path), "r") as io:
            series = io.read().acquisition["ElectricalSeries"]
            # The two channels have different gains, so those land in channel_conversion and must be
            # included — omitting it is off by a factor of 1e6.
            data = np.asarray(series.data)
            channel_conversion = np.asarray(series.channel_conversion)
            volts = data * channel_conversion * series.conversion + series.offset
            np.testing.assert_allclose(volts[:, 0] * 1e6, VALUES, rtol=A, atol=1e-9)
