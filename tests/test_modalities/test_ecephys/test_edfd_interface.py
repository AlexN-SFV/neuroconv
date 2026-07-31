"""
Tests for discontinuous EDF+ (``EDF+D``) support in the EDFRecordingInterface.

These build small synthetic EDF files byte by byte — including the ``EDF Annotations`` signal and its
per-record time-keeping TALs — so they run without external data, network access, or ``pyedflib``.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from pynwb import NWBHDF5IO

from neuroconv.datainterfaces import EDFRecordingInterface
from neuroconv.datainterfaces.ecephys.edf._edfd_extractor import EDFDRecordingExtractor
from neuroconv.datainterfaces.ecephys.edf._edfd_reader import (
    _parse_tals,
    _split_into_tals,
    edf_plus_header_fields,
    group_records_into_runs,
    is_discontinuous_edf,
    read_edf_header,
)

ANNOTATIONS_LABEL = "EDF Annotations"
NUMBER_OF_CHANNELS = 3
SAMPLES_PER_RECORD = 10
RECORD_DURATION = 1.0
SAMPLING_FREQUENCY = SAMPLES_PER_RECORD / RECORD_DURATION
CHANNEL_NAMES = ["ch0", "ch1", "ch2"]

# Records 0-3 contiguous, a 5 s gap, records 4-7 contiguous, a 2.5 s gap, records 8-11 contiguous.
GAPPED_ONSETS = [0.0, 1.0, 2.0, 3.0, 9.0, 10.0, 11.0, 12.0, 14.5, 15.5, 16.5, 17.5]
CONTIGUOUS_ONSETS = [index * RECORD_DURATION for index in range(12)]
EXPECTED_RUNS = [(0, 4, 0.0), (4, 4, 9.0), (8, 4, 14.5)]


def _field(value, width: int) -> bytes:
    text = str(value)
    assert len(text) <= width, f"header field {text!r} does not fit in {width} bytes"
    # latin-1, so a fixture can write the micro sign a real writer would put in a dimension field.
    return text.ljust(width).encode("latin-1")


def _record_index_of(onset: float, record_onsets, record_duration: float) -> int:
    for index, start in enumerate(record_onsets):
        if start <= onset < start + record_duration:
            return index
    return 0


def write_edf(
    path,
    *,
    record_onsets,
    data,
    reserved="EDF+D",
    record_duration=RECORD_DURATION,
    samples_per_record=SAMPLES_PER_RECORD,
    channel_names=tuple(CHANNEL_NAMES),
    dimensions=None,
    prefilters=None,
    physical_min=-1000.0,
    physical_max=1000.0,
    annotations=None,
    annotation_samples=32,
    include_annotations_signal=True,
    extra_annotations_signals=0,
    extra_annotations=None,
    starttime="12.35.13",
    recording_field="Startdate 16-SEP-2021 X X test",
    patient_field="X X X X",
    nul_pad_record_count=False,
    terminate_tals=True,
):
    """
    Write a minimal but spec-conforming EDF/EDF+ file.

    ``data`` is ``(n_records, n_channels, samples_per_record)`` of int16 digital values, and
    ``annotations`` is a list of ``(onset, duration_or_None, text)`` written into whichever record
    contains the onset.

    ``samples_per_record`` may be a list, one entry per data channel, to build a file whose signals do
    not share a sampling rate. ``extra_annotations_signals`` appends further ``EDF Annotations`` signals
    after the first one, which the spec permits.

    ``terminate_tals=False`` omits the NUL byte that ends each TAL, leaving only the NUL padding at
    the tail of the block, which is what Nihon Kohden's ``EDF+D`` export does.
    """
    channel_names = list(channel_names)
    number_of_channels = len(channel_names)
    number_of_records = len(record_onsets)
    if not isinstance(samples_per_record, (list, tuple)):
        samples_per_record = [samples_per_record] * number_of_channels
    samples_per_record = list(samples_per_record)
    dimensions = list(dimensions or ["uV"] * number_of_channels)
    prefilters = list(prefilters or [""] * number_of_channels)
    if not isinstance(physical_min, (list, tuple)):
        physical_min = [physical_min] * number_of_channels
    if not isinstance(physical_max, (list, tuple)):
        physical_max = [physical_max] * number_of_channels
    physical_min, physical_max = list(physical_min), list(physical_max)
    annotations = annotations or []

    labels = list(channel_names)
    samples = list(samples_per_record)
    if include_annotations_signal:
        labels.append(ANNOTATIONS_LABEL)
        samples.append(annotation_samples)
        for _ in range(extra_annotations_signals):
            labels.append(ANNOTATIONS_LABEL)
            samples.append(annotation_samples)
    number_of_signals = len(labels)
    header_size = 256 + number_of_signals * 256

    header = bytearray()
    header += _field(0, 8)
    header += _field(patient_field, 80)
    header += _field(recording_field, 80)
    header += _field("16.09.21", 8)
    header += _field(starttime, 8)
    header += _field(header_size, 8)
    header += _field(reserved, 44)
    if nul_pad_record_count:
        # Real writers sometimes NUL-pad numeric fields; this is what makes pyedflib reject a file.
        header += (str(number_of_records) + "\x00").ljust(8).encode("ascii")
    else:
        header += _field(number_of_records, 8)
    header += _field(record_duration, 8)
    header += _field(number_of_signals, 4)

    for label in labels:
        header += _field(label, 16)
    for _ in labels:
        header += _field("", 80)  # transducer
    for index in range(number_of_signals):
        header += _field(dimensions[index] if index < number_of_channels else "", 8)
    for index in range(number_of_signals):
        header += _field(physical_min[index] if index < number_of_channels else -1, 8)
    for index in range(number_of_signals):
        header += _field(physical_max[index] if index < number_of_channels else 1, 8)
    for _ in range(number_of_signals):
        header += _field(-32768, 8)
    for _ in range(number_of_signals):
        header += _field(32767, 8)
    for index in range(number_of_signals):
        header += _field(prefilters[index] if index < number_of_channels else "", 80)
    for index in range(number_of_signals):
        header += _field(samples[index], 8)
    for _ in range(number_of_signals):
        header += _field("", 32)  # per-signal reserved
    assert len(header) == header_size

    body = bytearray()
    for record_index in range(number_of_records):
        for channel_index in range(number_of_channels):
            channel_samples = np.asarray(data[record_index][channel_index], dtype="<i2")
            body += channel_samples[: samples_per_record[channel_index]].tobytes()
        if not include_annotations_signal:
            continue
        # The mandatory time-keeping TAL, then any real annotations belonging to this record.
        terminator = b"\x00" if terminate_tals else b""
        block = f"+{record_onsets[record_index]:.6f}".encode("ascii") + b"\x14\x14" + terminator
        for onset, duration, text in annotations:
            if record_index != _record_index_of(onset, record_onsets, record_duration):
                continue
            stamp = f"+{onset:.6f}".encode("ascii")
            if duration is not None:
                stamp += b"\x15" + f"{duration:g}".encode("ascii")
            block += stamp + b"\x14" + text.encode("utf-8") + b"\x14" + terminator
        assert len(block) <= annotation_samples * 2, "annotation block overflow in fixture"
        body += block.ljust(annotation_samples * 2, b"\x00")
        # Additional annotations signals carry no time-keeping TAL of their own, but the spec lets them
        # hold ordinary annotations — which is why writers add them — so the fixture puts one in each.
        for extra_index in range(extra_annotations_signals):
            extra = b""
            for onset, duration, text in extra_annotations or []:
                if record_index == _record_index_of(onset, record_onsets, record_duration):
                    extra += f"+{onset:.6f}".encode("ascii") + b"\x14" + text.encode("utf-8") + b"\x14\x00"
            assert len(extra) <= annotation_samples * 2, "extra annotation block overflow in fixture"
            body += extra.ljust(annotation_samples * 2, b"\x00")

    path = str(path)
    with open(path, "wb") as file:
        file.write(bytes(header))
        file.write(bytes(body))
    return path


@pytest.fixture
def digital_data():
    generator = np.random.default_rng(0)
    return generator.integers(-30000, 30000, size=(12, NUMBER_OF_CHANNELS, SAMPLES_PER_RECORD)).astype("int16")


def _expected_run_traces(digital_data, first_record, number_of_records):
    """The (n_samples, n_channels) digital values written for one run."""
    return np.concatenate([digital_data[first_record + offset].T for offset in range(number_of_records)], axis=0)


LOG_TRANSFORM_A = 0.002
LOG_TRANSFORM_YMIN = 0.01
LOG_TRANSFORM_PREFILTER = "sign*LN[sign*(uV      )/(0.01    )]/(0.002   )"

# Values spanning a range no 16-bit integer could hold, which is the point of the transform.
LOG_TRANSFORM_VALUES = [0.0, 0.05, 1.0, 250.0, 1e4, 5e6, -0.05, -1.0, -250.0, -1e4]


def encode_log_transform(values, a=LOG_TRANSFORM_A, ymin=LOG_TRANSFORM_YMIN):
    """
    Encode physical values as the integer counts an EDF+ "Filtered" signal stores.

    ``N = round(ln(Y/Ymin)/a)`` for ``Y > Ymin``, 0 for ``|Y| <= Ymin``, and the negative mirror below
    ``-Ymin``. See https://www.edfplus.info/specs/edffloat.html.
    """
    counts = np.zeros(len(values), dtype="int16")
    for index, value in enumerate(values):
        if value > ymin:
            counts[index] = round(np.log(value / ymin) / a)
        elif value < -ymin:
            counts[index] = -round(np.log(-value / ymin) / a)
    return counts


def write_log_transformed_edf(path, values, *, reserved="EDF+C", prefilter=LOG_TRANSFORM_PREFILTER):
    """One log-transformed signal beside one ordinary linear signal, which is the realistic mixture."""
    counts = encode_log_transform(values)
    samples_per_record = len(counts)
    linear = np.arange(samples_per_record, dtype="int16")
    data = [[counts, linear]]
    return (
        write_edf(
            path,
            record_onsets=[0.0],
            data=data,
            reserved=reserved,
            samples_per_record=samples_per_record,
            channel_names=["logch", "linch"],
            dimensions=["Filtered", "uV"],
            prefilters=[prefilter, ""],
            physical_min=[-32767.0, -1000.0],
            physical_max=[32767.0, 1000.0],
        ),
        counts,
        linear,
    )


class TestLogTransformedSignalsOnTheEDFDReader:
    """
    The EDF+D reader decodes transformed signals natively, using the shared implementation.

    The transformation itself — parsing, round-trip accuracy, the rejection paths — is covered once in
    test_edf_log_transform.py, which exercises it through the SpikeInterface path. What is specific here
    is that neuroconv's own reader reaches the same values.
    """

    # Only the two EDF+ variants: this fixture must carry an "EDF Annotations" signal, which EDF+
    # requires, and in a *plain* EDF neo treats that signal as ordinary data at its own sampling rate —
    # giving two streams and a file no reader would accept as one recording.
    @pytest.mark.parametrize("reserved, routed_here", [("EDF+D", True), ("EDF+C", False)])
    def test_only_discontinuity_routes_away_from_spikeinterface(self, tmp_path, reserved, routed_here):
        """
        Discontinuity is the only property SpikeInterface's reader cannot represent at all.

        A transformed signal only needs decoding on top of it, which the SpikeInterface path does by
        wrapping — so a *continuous* transformed file stays there, and both paths reach the same values
        through the shared implementation.
        """
        path, _, _ = write_log_transformed_edf(tmp_path / "log.edf", LOG_TRANSFORM_VALUES, reserved=reserved)
        interface = EDFRecordingInterface(file_path=path)

        assert interface.is_discontinuous is (reserved == "EDF+D")
        assert (type(interface.recording_extractor) is EDFDRecordingExtractor) is routed_here
        # Either way the transformed channel is reported and decoded.
        assert list(interface.log_transformed_channels) == ["logch"]
        traces = interface.recording_extractor.get_traces(segment_index=0)
        np.testing.assert_allclose(traces[:, 0], LOG_TRANSFORM_VALUES, rtol=LOG_TRANSFORM_A, atol=1e-12)

    def test_decoded_values_reach_the_recording(self, tmp_path):
        path, _, linear = write_log_transformed_edf(tmp_path / "log.edf", LOG_TRANSFORM_VALUES)
        recording = EDFDRecordingExtractor(file_path=path)

        # float64, not float32: the spec's own accuracy table reaches 1.4e68.
        assert recording.get_dtype() == np.dtype("float64")
        traces = recording.get_traces(segment_index=0)

        # The transform's gain is only the unit scaling and its offset is zero, since the exponential
        # relationship cannot be expressed as the scalar gain+offset an ElectricalSeries applies.
        assert recording.get_channel_gains()[0] == pytest.approx(1.0)  # uV -> uV
        assert recording.get_channel_offsets()[0] == 0.0
        np.testing.assert_allclose(traces[:, 0], LOG_TRANSFORM_VALUES, rtol=LOG_TRANSFORM_A, atol=1e-12)

        # The linear signal beside it keeps its raw digital values and its own gain.
        np.testing.assert_array_equal(traces[:, 1], linear)

    def test_chunked_and_subset_reads_decode_consistently(self, tmp_path):
        path, _, _ = write_log_transformed_edf(tmp_path / "log.edf", LOG_TRANSFORM_VALUES)
        recording = EDFDRecordingExtractor(file_path=path)
        full = recording.get_traces(segment_index=0)

        for start, end in [(0, 1), (2, 5), (4, len(LOG_TRANSFORM_VALUES))]:
            np.testing.assert_array_equal(
                recording.get_traces(segment_index=0, start_frame=start, end_frame=end), full[start:end]
            )
        # The transform must follow the channel, not its column position.
        np.testing.assert_array_equal(
            recording.get_traces(segment_index=0, channel_ids=["linch", "logch"]), full[:, [1, 0]]
        )
        np.testing.assert_array_equal(recording.get_traces(segment_index=0, channel_ids=["logch"]), full[:, [0]])

    def test_transformed_signal_can_be_skipped(self, tmp_path):
        """channels_to_skip is applied while reading, so an unreadable transform need not be fatal."""
        path, _, _ = write_log_transformed_edf(tmp_path / "log.edf", LOG_TRANSFORM_VALUES, prefilter="HP:0.1Hz")
        with pytest.raises(ValueError, match="logarithmically transformed"):
            EDFDRecordingExtractor(file_path=path)

        recording = EDFDRecordingExtractor(file_path=path, channels_to_skip=["logch"])
        assert list(recording.get_channel_ids()) == ["linch"]
        assert recording.get_dtype() == np.dtype("int16")


class TestEDFDReader:
    """The parsing layer, exercised directly."""

    def test_detects_discontinuous_from_reserved_field(self, tmp_path, digital_data):
        gapped = write_edf(tmp_path / "d.edf", record_onsets=GAPPED_ONSETS, data=digital_data, reserved="EDF+D")
        continuous = write_edf(tmp_path / "c.edf", record_onsets=CONTIGUOUS_ONSETS, data=digital_data, reserved="EDF+C")
        plain = write_edf(tmp_path / "p.edf", record_onsets=CONTIGUOUS_ONSETS, data=digital_data, reserved="")

        assert is_discontinuous_edf(file_path=gapped) is True
        assert is_discontinuous_edf(file_path=continuous) is False
        assert is_discontinuous_edf(file_path=plain) is False

    def test_tolerates_nul_padded_record_count(self, tmp_path, digital_data):
        """pyedflib rejects these files outright; the header must still parse."""
        path = write_edf(
            tmp_path / "nul.edf", record_onsets=GAPPED_ONSETS, data=digital_data, nul_pad_record_count=True
        )
        with open(path, "rb") as file:
            header = read_edf_header(file)
        assert header.number_of_records == 12
        assert header.is_discontinuous

    def test_recovers_start_datetime_from_recording_field(self, tmp_path, digital_data):
        path = write_edf(tmp_path / "d.edf", record_onsets=GAPPED_ONSETS, data=digital_data)
        with open(path, "rb") as file:
            header = read_edf_header(file)
        assert header.start_datetime == datetime(2021, 9, 16, 12, 35, 13)

    def test_falls_back_to_two_digit_year_with_clipping(self, tmp_path, digital_data):
        path = write_edf(tmp_path / "d.edf", record_onsets=GAPPED_ONSETS, data=digital_data, recording_field="X X X X")
        with open(path, "rb") as file:
            header = read_edf_header(file)
        # "16.09.21" -> 2021 under the EDF clipping rule (00-84 mean 2000-2084).
        assert header.start_datetime == datetime(2021, 9, 16, 12, 35, 13)

    def test_number_of_records_recovered_when_unknown(self, tmp_path, digital_data):
        path = write_edf(tmp_path / "d.edf", record_onsets=GAPPED_ONSETS, data=digital_data)
        # Overwrite the record count with -1, as a writer does while a file is still open.
        with open(path, "r+b") as file:
            file.seek(236)
            file.write(_field(-1, 8))
        with open(path, "rb") as file:
            header = read_edf_header(file)
        assert header.number_of_records == 12

    def test_groups_records_into_runs(self):
        runs = group_records_into_runs(
            onsets=np.array(GAPPED_ONSETS),
            samples_per_record=SAMPLES_PER_RECORD,
            sampling_frequency=SAMPLING_FREQUENCY,
        )
        assert runs == EXPECTED_RUNS

    def test_backward_jump_starts_a_new_run(self):
        runs = group_records_into_runs(
            onsets=np.array([0.0, 1.0, 0.5, 1.5]),
            samples_per_record=SAMPLES_PER_RECORD,
            sampling_frequency=SAMPLING_FREQUENCY,
        )
        assert runs == [(0, 2, 0.0), (2, 2, 0.5)]

    def test_rounded_onsets_do_not_create_spurious_runs(self):
        """
        A record duration that does not terminate in the printed decimals must stay one run.

        Onsets are ASCII decimals, so a writer printing three decimals for a 1/3 s record duration
        yields inter-record distances of 0.334 and 0.333 s. Comparing in seconds would call those gaps
        and shatter a continuous recording; comparing in whole samples does not.
        """
        sampling_frequency = 300.0
        samples_per_record = 100  # 1/3 s per record
        onsets = np.array([round(index / 3.0, 3) for index in range(60)])
        runs = group_records_into_runs(
            onsets=onsets, samples_per_record=samples_per_record, sampling_frequency=sampling_frequency
        )
        assert runs == [(0, 60, 0.0)]

    def test_single_sample_gap_is_still_detected(self):
        """The rounding must not be so coarse that a real one-sample gap is missed."""
        onsets = np.array([0.0, 1.0, 2.1, 3.1])
        runs = group_records_into_runs(
            onsets=onsets, samples_per_record=SAMPLES_PER_RECORD, sampling_frequency=SAMPLING_FREQUENCY
        )
        assert runs == [(0, 2, 0.0), (2, 2, 2.1)]

    def test_traces_match_written_digital_values(self, tmp_path, digital_data):
        path = write_edf(tmp_path / "d.edf", record_onsets=GAPPED_ONSETS, data=digital_data)
        recording = EDFDRecordingExtractor(file_path=path)

        assert recording.get_num_segments() == 3
        for segment_index, (first_record, number_of_records, t_start) in enumerate(EXPECTED_RUNS):
            expected = _expected_run_traces(digital_data, first_record, number_of_records)
            np.testing.assert_array_equal(recording.get_traces(segment_index=segment_index), expected)
            assert recording._recording_segments[segment_index].t_start == t_start

    @pytest.mark.parametrize("start_frame, end_frame", [(0, 1), (3, 7), (9, 11), (10, 20), (37, 40), (0, 40)])
    def test_chunked_reads_match_full_read(self, tmp_path, digital_data, start_frame, end_frame):
        """Chunk boundaries must not fall foul of the record-to-frame arithmetic."""
        path = write_edf(tmp_path / "d.edf", record_onsets=GAPPED_ONSETS, data=digital_data)
        recording = EDFDRecordingExtractor(file_path=path)
        expected = _expected_run_traces(digital_data, 4, 4)
        traces = recording.get_traces(segment_index=1, start_frame=start_frame, end_frame=end_frame)
        np.testing.assert_array_equal(traces, expected[start_frame:end_frame])

    @pytest.mark.parametrize("channel_ids", [["ch1"], ["ch0", "ch2"], ["ch2", "ch0"]])
    def test_channel_subset_reads(self, tmp_path, digital_data, channel_ids):
        path = write_edf(tmp_path / "d.edf", record_onsets=GAPPED_ONSETS, data=digital_data)
        recording = EDFDRecordingExtractor(file_path=path)
        expected = _expected_run_traces(digital_data, 4, 4)
        columns = [CHANNEL_NAMES.index(channel_id) for channel_id in channel_ids]
        traces = recording.get_traces(segment_index=1, start_frame=7, end_frame=23, channel_ids=channel_ids)
        np.testing.assert_array_equal(traces, expected[7:23][:, columns])

    def test_annotations_signal_is_not_a_data_channel(self, tmp_path, digital_data):
        path = write_edf(tmp_path / "d.edf", record_onsets=GAPPED_ONSETS, data=digital_data)
        recording = EDFDRecordingExtractor(file_path=path)
        assert list(recording.get_channel_ids()) == CHANNEL_NAMES
        assert ANNOTATIONS_LABEL not in recording.get_channel_ids()

    def test_annotations_are_parsed(self, tmp_path, digital_data):
        annotations = [(2.5, None, "Marker: 601"), (10.25, 1.5, "seizure onset")]
        path = write_edf(tmp_path / "d.edf", record_onsets=GAPPED_ONSETS, data=digital_data, annotations=annotations)
        recording = EDFDRecordingExtractor(file_path=path)
        assert recording.annotations_from_file == [
            dict(onset=2.5, duration=None, text="Marker: 601"),
            dict(onset=10.25, duration=1.5, text="seizure onset"),
        ]

    def test_heterogeneous_sampling_rates_raise(self, tmp_path, digital_data):
        """A file whose signals have different rates is rejected, naming the remedy."""
        path = write_edf(
            tmp_path / "rates.edf",
            record_onsets=CONTIGUOUS_ONSETS,
            data=digital_data,
            samples_per_record=[SAMPLES_PER_RECORD, SAMPLES_PER_RECORD, SAMPLES_PER_RECORD // 2],
        )
        with pytest.raises(ValueError, match="differing sampling rates"):
            EDFDRecordingExtractor(file_path=path)

    def test_off_rate_channel_can_be_skipped(self, tmp_path, digital_data):
        """Which is what the error above tells the user to do."""
        path = write_edf(
            tmp_path / "rates.edf",
            record_onsets=CONTIGUOUS_ONSETS,
            data=digital_data,
            samples_per_record=[SAMPLES_PER_RECORD, SAMPLES_PER_RECORD, SAMPLES_PER_RECORD // 2],
        )
        recording = EDFDRecordingExtractor(file_path=path, channels_to_skip=["ch2"])
        assert list(recording.get_channel_ids()) == ["ch0", "ch1"]

    def test_discontinuous_without_annotations_signal_raises(self, tmp_path, digital_data):
        path = write_edf(
            tmp_path / "no_annot.edf",
            record_onsets=GAPPED_ONSETS,
            data=digital_data,
            include_annotations_signal=False,
        )
        with pytest.raises(ValueError, match="no 'EDF Annotations' signal"):
            EDFDRecordingExtractor(file_path=path)

    def test_no_channels_left_raises(self, tmp_path, digital_data):
        path = write_edf(tmp_path / "d.edf", record_onsets=GAPPED_ONSETS, data=digital_data)
        with pytest.raises(ValueError, match="No data channels remain"):
            EDFDRecordingExtractor(file_path=path, channels_to_skip=CHANNEL_NAMES)

    def test_symmetric_digital_range_gives_zero_offset(self, tmp_path, digital_data):
        """
        The gain/offset convention must match neo's, i.e. the EDF/EDF+C path.

        Dividing the physical range by ``digital_max - digital_min`` (the spec's endpoint mapping, and
        what pyedflib does) leaves half an LSB of DC that scales with each channel's physical range. A
        file mixing ranges — EEG in uV beside a wider-range auxiliary channel, the ordinary clinical
        layout — then gets heterogeneous offsets, and a single ElectricalSeries holds only one scalar
        offset, so the same recording would convert as EDF+C and fail as EDF+D.
        """
        path = write_edf(
            tmp_path / "mixed.edf",
            record_onsets=CONTIGUOUS_ONSETS,
            data=digital_data,
            physical_min=[-3200.0, -3200.0, -10000.0],
            physical_max=[3200.0, 3200.0, 10000.0],
        )
        recording = EDFDRecordingExtractor(file_path=path)
        np.testing.assert_allclose(recording.get_channel_offsets(), [0.0, 0.0, 0.0])
        # And the gains match neo's convention exactly, not the endpoint mapping.
        expected_gains = [6400.0 / 65536, 6400.0 / 65536, 20000.0 / 65536]
        np.testing.assert_allclose(recording.get_channel_gains(), expected_gains)

    def test_every_annotations_signal_is_excluded_from_data(self, tmp_path, digital_data):
        """The spec permits additional 'EDF Annotations' signals; none may become a data channel."""
        path = write_edf(
            tmp_path / "two_annot.edf",
            record_onsets=GAPPED_ONSETS,
            data=digital_data,
            extra_annotations_signals=1,
        )
        with open(path, "rb") as file:
            header = read_edf_header(file)
        assert len(header.annotations_signal_indices) == 2

        recording = EDFDRecordingExtractor(file_path=path)
        assert list(recording.get_channel_ids()) == CHANNEL_NAMES
        # And the run layout is still recovered from the first annotations signal.
        assert recording.runs == EXPECTED_RUNS

    def test_unreadable_time_keeping_onset_raises_for_discontinuous(self, tmp_path, digital_data):
        """
        Assuming contiguity when an onset cannot be read is the one thing EDF+D exists to rule out.

        A comma decimal separator is a real EDF pathology; it must not silently place every subsequent
        sample at the wrong time.
        """
        path = write_edf(tmp_path / "d.edf", record_onsets=GAPPED_ONSETS, data=digital_data)
        raw = Path(path).read_bytes().replace(b"+9.000000", b"+9,000000")
        Path(path).write_bytes(raw)
        with pytest.raises(ValueError, match="no readable time-keeping annotation"):
            EDFDRecordingExtractor(file_path=path)

    def test_leading_annotation_is_not_mistaken_for_the_time_keeping_tal(self, tmp_path, digital_data):
        """Only an empty-text leading TAL is the time-keeping one; a real annotation must not be used."""
        path = write_edf(tmp_path / "d.edf", record_onsets=GAPPED_ONSETS, data=digital_data)
        # Replace record 0's time-keeping TAL (+0.000000[20][20][0]) with a *texted* TAL at the same onset.
        raw = Path(path).read_bytes()
        assert raw.count(b"+0.000000\x14\x14\x00") == 1
        Path(path).write_bytes(raw.replace(b"+0.000000\x14\x14\x00", b"+0.000000\x14seizure\x14\x00", 1))
        with pytest.raises(ValueError, match="no readable time-keeping annotation"):
            EDFDRecordingExtractor(file_path=path)

    def test_unterminated_tals_are_split_apart(self, tmp_path, digital_data):
        """
        Some writers never NUL-terminate a TAL and only NUL-pad the tail of the annotations block, so a
        record's time-keeping TAL and its real annotations arrive as one NUL-free run of bytes.

        Nihon Kohden's EDF+D export does this. Reading the run as a single TAL makes the time-keeping
        annotation look texted, which used to make every annotated record's onset unrecoverable and so
        rejected an otherwise perfectly readable file.
        """
        annotations = [(2.5, None, "Segment: REC START"), (10.25, 1.5, "seizure onset")]
        path = write_edf(
            tmp_path / "d.edf",
            record_onsets=GAPPED_ONSETS,
            data=digital_data,
            annotations=annotations,
            terminate_tals=False,
        )
        assert b"\x14\x14\x00" not in Path(path).read_bytes()[36352:], "fixture still terminates its TALs"

        recording = EDFDRecordingExtractor(file_path=path)
        assert recording.runs == EXPECTED_RUNS
        assert recording.annotations_from_file == [
            dict(onset=2.5, duration=None, text="Segment: REC START"),
            dict(onset=10.25, duration=1.5, text="seizure onset"),
        ]

    def test_unterminated_tals_give_the_same_result_as_terminated_ones(self, tmp_path, digital_data):
        """The NUL terminator is the writer's business; neither spelling may change what is read."""
        annotations = [(2.5, None, "Marker: 601")]
        paths = {
            terminated: write_edf(
                tmp_path / f"{terminated}.edf",
                record_onsets=GAPPED_ONSETS,
                data=digital_data,
                annotations=annotations,
                terminate_tals=terminated,
            )
            for terminated in (True, False)
        }
        recordings = {key: EDFDRecordingExtractor(file_path=path) for key, path in paths.items()}
        assert recordings[True].runs == recordings[False].runs
        assert recordings[True].annotations_from_file == recordings[False].annotations_from_file
        np.testing.assert_array_equal(
            recordings[True].get_traces(segment_index=1), recordings[False].get_traces(segment_index=1)
        )

    def test_numeric_annotation_text_does_not_start_a_new_tal(self):
        """
        An EDF+ onset always carries an explicit sign, which is what lets an unterminated TAL be split.

        A bare number such as a stimulus code is an annotation text, not the next TAL's onset — the
        real recording that motivated this uses ``601`` as an event label.
        """
        assert _split_into_tals(b"+0.868000\x14\x14+1.000000\x14601\x14") == [
            [b"+0.868000", b""],
            [b"+1.000000", b"601", b""],
        ]
        assert _parse_tals(b"+0.868000\x14\x14+1.000000\x14601\x14") == [
            (0.868, None, []),
            (1.0, None, ["601"]),
        ]

    def test_multiple_annotations_in_one_tal_stay_in_one_tal(self):
        """The spec lets a single TAL carry several texts; splitting must not break that shape."""
        assert _parse_tals(b"+180\x14Lights off\x14Close door\x14\x00") == [(180.0, None, ["Lights off", "Close door"])]

    def test_junk_after_the_time_keeping_tal_does_not_cost_the_onset(self):
        """
        The record's position is worth more than an unreadable annotation.

        Splitting at the time-keeping annotation's empty text means unparsable trailing bytes lose only
        themselves, where previously they took the whole file down with them.
        """
        assert _parse_tals(b"+3.000000\x14\x14not-a-tal\x14") == [(3.0, None, [])]

    def test_onset_between_readable_neighbours_is_recovered_with_a_warning(self, tmp_path, digital_data):
        """
        A record bracketed by onsets that span exactly the records in between cannot be hiding a gap, so
        its start time follows from its neighbours and the file need not be rejected.
        """
        path = write_edf(tmp_path / "d.edf", record_onsets=GAPPED_ONSETS, data=digital_data)
        # Record 2 sits inside the first run, so records 1 and 3 bracket it one record duration apart.
        raw = Path(path).read_bytes()
        assert raw.count(b"+2.000000") == 1
        Path(path).write_bytes(raw.replace(b"+2.000000", b"+2,000000", 1))

        with pytest.warns(UserWarning, match="follow from their neighbours"):
            recording = EDFDRecordingExtractor(file_path=path)
        assert recording.runs == EXPECTED_RUNS

    def test_unrecoverable_onset_error_shows_the_offending_bytes(self, tmp_path, digital_data):
        """
        The old message could only guess at the cause, which sent this exact investigation down the
        wrong path. Printing the block lets a reader see what the writer actually emitted.
        """
        path = write_edf(tmp_path / "d.edf", record_onsets=GAPPED_ONSETS, data=digital_data)
        raw = Path(path).read_bytes()
        Path(path).write_bytes(raw.replace(b"+9.000000", b"+9,000000", 1))
        with pytest.raises(ValueError, match=r"block of record 4 reads b'\+9,000000"):
            EDFDRecordingExtractor(file_path=path)

    def test_truncated_final_record_is_dropped_not_read(self, tmp_path, digital_data):
        """An interrupted acquisition must not defer an opaque reshape error to read time."""
        path = write_edf(tmp_path / "d.edf", record_onsets=CONTIGUOUS_ONSETS, data=digital_data)
        full = Path(path).read_bytes()
        with open(path, "rb") as file:
            header = read_edf_header(file)
        Path(path).write_bytes(full[: -header.record_size_bytes // 2])  # half a record short

        recording = EDFDRecordingExtractor(file_path=path)
        assert recording.get_num_samples(0) == 11 * SAMPLES_PER_RECORD
        # And the whole recording is readable, rather than blowing up on the partial record.
        assert recording.get_traces(segment_index=0).shape == (11 * SAMPLES_PER_RECORD, NUMBER_OF_CHANNELS)

    def test_zero_samples_per_record_is_named_not_a_zero_division(self, tmp_path, digital_data):
        """
        The record arithmetic divides by the record size, and that now runs for every EDF file.

        The routing decision parses the header before SpikeInterface is consulted, so an unguarded
        division would show a bare ZeroDivisionError to someone converting a degenerate *plain* EDF.
        """
        path = write_edf(
            tmp_path / "zero.edf",
            record_onsets=GAPPED_ONSETS,
            data=digital_data,
            samples_per_record=[0] * NUMBER_OF_CHANNELS,
            include_annotations_signal=False,
        )
        with pytest.raises(ValueError, match="zero samples per data record"):
            EDFDRecordingExtractor(file_path=path)

    def test_wrong_header_size_is_rejected(self, tmp_path, digital_data):
        """A header size disagreeing with 256*(n_signals+1) means header padding would be read as data."""
        path = write_edf(tmp_path / "d.edf", record_onsets=GAPPED_ONSETS, data=digital_data)
        with open(path, "r+b") as file:
            file.seek(184)
            file.write(_field(9999, 8))
        with pytest.raises(ValueError, match="header size"):
            EDFDRecordingExtractor(file_path=path)

    def test_duplicate_and_blank_labels_are_rejected(self, tmp_path, digital_data):
        duplicate = write_edf(
            tmp_path / "dup.edf", record_onsets=GAPPED_ONSETS, data=digital_data, channel_names=["a", "a", "b"]
        )
        with pytest.raises(ValueError, match="duplicate channel labels"):
            EDFDRecordingExtractor(file_path=duplicate)

        blank = write_edf(
            tmp_path / "blank.edf", record_onsets=GAPPED_ONSETS, data=digital_data, channel_names=["a", "", "b"]
        )
        with pytest.raises(ValueError, match="blank label"):
            EDFDRecordingExtractor(file_path=blank)

    def test_unknown_channels_to_skip_is_rejected(self, tmp_path, digital_data):
        """The SpikeInterface path's remove_channels raises; silently ignoring a typo would not."""
        path = write_edf(tmp_path / "d.edf", record_onsets=GAPPED_ONSETS, data=digital_data)
        with pytest.raises(ValueError, match="not in this file"):
            EDFDRecordingExtractor(file_path=path, channels_to_skip=["nonexistent"])

    def test_malformed_header_field_names_the_field(self, tmp_path, digital_data):
        path = write_edf(tmp_path / "d.edf", record_onsets=GAPPED_ONSETS, data=digital_data)
        with open(path, "r+b") as file:
            file.seek(244)  # duration of a data record
            file.write(_field("abc", 8))
        with pytest.raises(ValueError, match="duration of a data record"):
            EDFDRecordingExtractor(file_path=path)

    def test_start_datetime_falls_back_when_day_is_zeroed(self, tmp_path, digital_data):
        """Day-level de-identification ("00-JAN-2021") parses but cannot be assembled."""
        path = write_edf(
            tmp_path / "d.edf",
            record_onsets=GAPPED_ONSETS,
            data=digital_data,
            recording_field="Startdate 00-JAN-2021 X X test",
        )
        with open(path, "rb") as file:
            header = read_edf_header(file)
        # Falls back to the header's own startdate field, which is intact.
        assert header.start_datetime == datetime(2021, 9, 16, 12, 35, 13)

    def test_extractor_is_picklable_and_round_trips_through_to_dict(self, tmp_path, digital_data):
        """Needed for recording.save() and any n_jobs > 1 path."""
        import importlib
        import pickle

        path = write_edf(tmp_path / "d.edf", record_onsets=GAPPED_ONSETS, data=digital_data)
        recording = EDFDRecordingExtractor(file_path=path)

        unpickled = pickle.loads(pickle.dumps(recording))
        assert unpickled.get_num_segments() == recording.get_num_segments()
        np.testing.assert_array_equal(unpickled.get_traces(segment_index=1), recording.get_traces(segment_index=1))

        # to_dict records a resolvable import path, and the dict reloads. Both need the classes to
        # live at module scope, and reloading additionally needs ``neuroconv.__version__`` to exist,
        # because SpikeInterface stamps and re-reads the defining package's version.
        from spikeinterface.core import load

        dictionary = recording.to_dict()
        assert dictionary["class"].endswith("_edfd_extractor.EDFDRecordingExtractor")
        module_name, class_name = dictionary["class"].rsplit(".", 1)
        assert getattr(importlib.import_module(module_name), class_name) is EDFDRecordingExtractor

        reloaded = load(dictionary)
        assert reloaded.get_num_segments() == recording.get_num_segments()
        np.testing.assert_array_equal(reloaded.get_traces(segment_index=2), recording.get_traces(segment_index=2))

    def test_header_fields_mapped_to_pyedflib_shape(self, tmp_path, digital_data):
        path = write_edf(
            tmp_path / "d.edf",
            record_onsets=GAPPED_ONSETS,
            data=digital_data,
            patient_field="MCH-42 F 02-MAY-1951 Haagse_Harry",
            recording_field="Startdate 16-SEP-2021 PSG-1 Tech7 BESA_export",
        )
        with open(path, "rb") as file:
            fields = edf_plus_header_fields(read_edf_header(file))
        assert fields["patientcode"] == "MCH-42"
        assert fields["sex"] == "F"
        assert fields["birthdate"] == "02-MAY-1951"
        assert fields["technician"] == "Tech7"
        assert fields["equipment"] == "BESA_export"
        assert fields["startdate"] == datetime(2021, 9, 16, 12, 35, 13)

    def test_unknown_subfields_become_empty(self, tmp_path, digital_data):
        """EDF+ writes a bare 'X' for unknown subfields; it must not leak into metadata."""
        path = write_edf(tmp_path / "d.edf", record_onsets=GAPPED_ONSETS, data=digital_data)
        with open(path, "rb") as file:
            fields = edf_plus_header_fields(read_edf_header(file))
        assert fields["patientcode"] == ""
        assert fields["technician"] == ""


class TestEDFDRecordingInterface:
    """The interface layer, including what reaches the NWB file."""

    def test_continuous_file_still_uses_spikeinterface(self, tmp_path, digital_data):
        """Regression: plain EDF+C must keep going through the existing reader, untouched."""
        pytest.importorskip("pyedflib")
        path = write_edf(tmp_path / "c.edf", record_onsets=CONTIGUOUS_ONSETS, data=digital_data, reserved="EDF+C")
        interface = EDFRecordingInterface(file_path=path)
        assert interface.is_discontinuous is False
        assert interface.number_of_runs == 1
        assert type(interface.recording_extractor).__name__ == "EDFRecordingExtractor"

    def test_discontinuous_file_uses_the_edfd_reader(self, tmp_path, digital_data):
        path = write_edf(tmp_path / "d.edf", record_onsets=GAPPED_ONSETS, data=digital_data)
        interface = EDFRecordingInterface(file_path=path)
        assert interface.is_discontinuous is True
        assert interface.number_of_runs == 3
        assert interface.recording_extractor.get_num_segments() == 3

    def test_labelled_discontinuous_but_contiguous_is_a_single_series(self, tmp_path, digital_data):
        """
        The common real-world case: exporters label continuous recordings EDF+D.

        Such a file must convert exactly like a plain EDF — one series, no epochs table.
        """
        path = write_edf(tmp_path / "fake_d.edf", record_onsets=CONTIGUOUS_ONSETS, data=digital_data, reserved="EDF+D")
        interface = EDFRecordingInterface(file_path=path)
        assert interface.is_discontinuous is True
        assert interface.number_of_runs == 1

        nwbfile_path = tmp_path / "fake_d.nwb"
        interface.run_conversion(nwbfile_path=str(nwbfile_path), metadata=_metadata(interface), overwrite=True)
        with NWBHDF5IO(str(nwbfile_path), "r") as io:
            nwbfile = io.read()
            assert sorted(nwbfile.acquisition) == ["ElectricalSeries"]
            assert "runs" not in (nwbfile.intervals or {})

    def test_gapped_file_writes_one_series_per_run(self, tmp_path, digital_data):
        path = write_edf(tmp_path / "d.edf", record_onsets=GAPPED_ONSETS, data=digital_data)
        interface = EDFRecordingInterface(file_path=path)

        nwbfile_path = tmp_path / "d.nwb"
        interface.run_conversion(nwbfile_path=str(nwbfile_path), metadata=_metadata(interface), overwrite=True)

        with NWBHDF5IO(str(nwbfile_path), "r") as io:
            nwbfile = io.read()
            names = sorted(nwbfile.acquisition)
            assert names == ["ElectricalSeries0", "ElectricalSeries1", "ElectricalSeries2"]

            # One shared electrodes table, not one per run.
            assert len(nwbfile.electrodes) == NUMBER_OF_CHANNELS

            for name, (first_record, number_of_records, t_start) in zip(names, EXPECTED_RUNS):
                series = nwbfile.acquisition[name]
                assert series.starting_time == pytest.approx(t_start)
                assert series.rate == pytest.approx(SAMPLING_FREQUENCY)
                expected = _expected_run_traces(digital_data, first_record, number_of_records)
                np.testing.assert_array_equal(np.asarray(series.data), expected)

    def test_gapped_file_writes_runs_table(self, tmp_path, digital_data):
        path = write_edf(tmp_path / "d.edf", record_onsets=GAPPED_ONSETS, data=digital_data)
        interface = EDFRecordingInterface(file_path=path)
        nwbfile_path = tmp_path / "d.nwb"
        interface.run_conversion(nwbfile_path=str(nwbfile_path), metadata=_metadata(interface), overwrite=True)

        with NWBHDF5IO(str(nwbfile_path), "r") as io:
            nwbfile = io.read()
            # A dedicated table, not nwbfile.epochs, which is a shared singleton another interface may
            # already have populated.
            assert nwbfile.epochs is None
            runs = nwbfile.intervals["runs"].to_dataframe()
            assert list(runs["run_index"]) == [0, 1, 2]
            assert list(runs["start_time"]) == pytest.approx([0.0, 9.0, 14.5])
            assert list(runs["stop_time"]) == pytest.approx([4.0, 13.0, 18.5])

    def test_annotations_written_to_time_intervals(self, tmp_path, digital_data):
        annotations = [(2.5, None, "Marker: 601"), (10.25, 1.5, "seizure onset")]
        path = write_edf(tmp_path / "d.edf", record_onsets=GAPPED_ONSETS, data=digital_data, annotations=annotations)
        interface = EDFRecordingInterface(file_path=path)
        nwbfile_path = tmp_path / "d.nwb"
        interface.run_conversion(nwbfile_path=str(nwbfile_path), metadata=_metadata(interface), overwrite=True)

        with NWBHDF5IO(str(nwbfile_path), "r") as io:
            table = io.read().intervals["annotations"].to_dataframe()
            assert list(table["label"]) == ["Marker: 601", "seizure onset"]
            assert list(table["start_time"]) == pytest.approx([2.5, 10.25])
            # A duration-less annotation gets a zero-length interval.
            assert list(table["stop_time"]) == pytest.approx([2.5, 11.75])

    def test_write_annotations_and_epochs_can_be_disabled(self, tmp_path, digital_data):
        annotations = [(2.5, None, "Marker: 601")]
        path = write_edf(tmp_path / "d.edf", record_onsets=GAPPED_ONSETS, data=digital_data, annotations=annotations)
        interface = EDFRecordingInterface(file_path=path)
        nwbfile_path = tmp_path / "d.nwb"
        interface.run_conversion(
            nwbfile_path=str(nwbfile_path),
            metadata=_metadata(interface),
            overwrite=True,
            write_annotations=False,
            write_runs=False,
        )
        with NWBHDF5IO(str(nwbfile_path), "r") as io:
            nwbfile = io.read()
            assert "annotations" not in (nwbfile.intervals or {})
            assert "runs" not in (nwbfile.intervals or {})

    def test_conversion_options_schema_keeps_inherited_options(self):
        """A **conversion_options catch-all would erase the inherited options from the schema."""
        schema = EDFRecordingInterface.get_conversion_options_schema(EDFRecordingInterface)
        assert {"write_annotations", "write_runs"} <= set(schema["properties"])
        assert {"stub_test", "iterator_type", "iterator_options", "write_electrical_series"} <= set(
            schema["properties"]
        )
        # Permissive validation would let typo'd option names through silently.
        assert schema["additionalProperties"] is False

    def test_channels_to_skip(self, tmp_path, digital_data):
        path = write_edf(tmp_path / "d.edf", record_onsets=GAPPED_ONSETS, data=digital_data)
        interface = EDFRecordingInterface(file_path=path, channels_to_skip=["ch1"])
        assert list(interface.recording_extractor.get_channel_ids()) == ["ch0", "ch2"]

    def test_get_available_channel_ids_excludes_annotations_signal(self, tmp_path, digital_data):
        """It must match the EDF+C path, which never exposes the annotations signal as a channel."""
        path = write_edf(tmp_path / "d.edf", record_onsets=GAPPED_ONSETS, data=digital_data)
        channel_ids = EDFRecordingInterface.get_available_channel_ids(file_path=path)
        assert channel_ids == CHANNEL_NAMES

    def test_metadata_from_header(self, tmp_path, digital_data):
        path = write_edf(
            tmp_path / "d.edf",
            record_onsets=GAPPED_ONSETS,
            data=digital_data,
            patient_field="MCH-42 F 02-MAY-1951 Haagse_Harry",
            recording_field="Startdate 16-SEP-2021 PSG-1 Tech7 BESA_export",
        )
        interface = EDFRecordingInterface(file_path=path)
        metadata = interface.get_metadata()
        assert metadata["NWBFile"]["session_start_time"] == datetime(2021, 9, 16, 12, 35, 13)
        assert metadata["NWBFile"]["experimenter"] == "Tech7"
        assert metadata["Subject"]["subject_id"] == "MCH-42"
        assert metadata["Subject"]["date_of_birth"] == datetime(1951, 5, 2)
        # Schema-required fields are filled in so the metadata stays valid.
        assert metadata["Subject"]["species"] == "Unknown species"
        # The patient field states "F", and edf_plus_header_fields surfaces it under the same "sex" key
        # pyedflib uses, so the EDF+D path feeds the shared extraction exactly as the SpikeInterface one.
        assert metadata["Subject"]["sex"] == "F"

    def test_unparseable_birthdate_is_dropped(self, tmp_path, digital_data):
        """A date pynwb could not accept must not be forwarded to it."""
        path = write_edf(
            tmp_path / "d.edf",
            record_onsets=GAPPED_ONSETS,
            data=digital_data,
            patient_field="MCH-42 F not-a-date Harry",
        )
        metadata = EDFRecordingInterface(file_path=path).get_metadata()
        assert metadata["Subject"]["subject_id"] == "MCH-42"
        assert "date_of_birth" not in metadata["Subject"]

    def test_tables_follow_temporal_alignment(self, tmp_path, digital_data):
        """
        The annotations and runs tables must move with the data when the recording is aligned.

        Syncing a clinical recording to a behavioural stream is the normal reason to touch this API, and
        table times derived from the raw file onsets would then point at the wrong moment.
        """
        path = write_edf(
            tmp_path / "d.edf",
            record_onsets=GAPPED_ONSETS,
            data=digital_data,
            annotations=[(2.5, None, "Marker: 601")],
        )
        interface = EDFRecordingInterface(file_path=path)
        interface.set_aligned_starting_time(aligned_starting_time=100.0)

        nwbfile_path = tmp_path / "d.nwb"
        interface.run_conversion(nwbfile_path=str(nwbfile_path), metadata=_metadata(interface), overwrite=True)

        with NWBHDF5IO(str(nwbfile_path), "r") as io:
            nwbfile = io.read()
            assert nwbfile.acquisition["ElectricalSeries0"].starting_time == pytest.approx(100.0)
            assert list(nwbfile.intervals["annotations"].to_dataframe()["start_time"]) == pytest.approx([102.5])
            assert list(nwbfile.intervals["runs"].to_dataframe()["start_time"]) == pytest.approx([100.0, 109.0, 114.5])

    def test_tables_follow_per_segment_alignment(self, tmp_path, digital_data):
        """
        Each run and annotation must move by *its own* segment's delta, not by segment 0's.

        ``set_aligned_segment_starting_times`` is the natural API for a multi-run recording — runs
        sitting at different times being the whole premise of EDF+D — and a single shared delta leaves
        every later run early by however much its own segment moved.
        """
        annotations = [(2.5, None, "in run 0"), (10.25, None, "in run 1"), (16.0, None, "in run 2")]
        path = write_edf(tmp_path / "d.edf", record_onsets=GAPPED_ONSETS, data=digital_data, annotations=annotations)
        interface = EDFRecordingInterface(file_path=path)
        interface.set_aligned_segment_starting_times(aligned_segment_starting_times=[100.0, 500.0, 900.0])

        nwbfile_path = tmp_path / "d.nwb"
        interface.run_conversion(nwbfile_path=str(nwbfile_path), metadata=_metadata(interface), overwrite=True)

        with NWBHDF5IO(str(nwbfile_path), "r") as io:
            nwbfile = io.read()
            series_starts = [nwbfile.acquisition[name].starting_time for name in sorted(nwbfile.acquisition)]
            assert series_starts == pytest.approx([100.0, 509.0, 914.5])

            # Each run tracks its own series, rather than all three sharing segment 0's delta.
            runs = nwbfile.intervals["runs"].to_dataframe()
            assert list(runs["start_time"]) == pytest.approx([100.0, 509.0, 914.5])

            # And each annotation moves with the run that contains it: +100, +500, +900 respectively.
            annotation_starts = list(nwbfile.intervals["annotations"].to_dataframe()["start_time"])
            assert annotation_starts == pytest.approx([102.5, 510.25, 916.0])

    def test_annotations_are_collected_from_every_annotations_signal(self, tmp_path, digital_data):
        """Exclusion from the data channels was pinned; collection from the extra signals was not."""
        path = write_edf(
            tmp_path / "two_annot.edf",
            record_onsets=GAPPED_ONSETS,
            data=digital_data,
            annotations=[(2.5, None, "from the first signal")],
            extra_annotations_signals=1,
            extra_annotations=[(10.25, None, "from the second signal")],
        )
        recording = EDFDRecordingExtractor(file_path=path)

        texts = sorted(annotation["text"] for annotation in recording.annotations_from_file)
        assert texts == ["from the first signal", "from the second signal"]

    def test_non_voltage_channel_warns_and_is_read_as_microvolts(self, tmp_path, digital_data):
        """
        A non-voltage channel is kept, assumed to be microvolts, and reported.

        Microvolts is the assumption least likely to be wrong by a large factor — and, unlike the
        "assume volts" that reusing the shared unit helper produced, it cannot inflate a value by 1e6.
        The warning names the unit and points at channels_to_skip and EDFAnalogInterface.
        """
        path = write_edf(
            tmp_path / "spo2.edf",
            record_onsets=GAPPED_ONSETS,
            data=digital_data,
            channel_names=["ch0", "ch1", "SpO2"],
            dimensions=["uV", "uV", "%"],
        )
        with pytest.warns(UserWarning, match="not a voltage"):
            recording = EDFDRecordingExtractor(file_path=path)

        # Kept, and scaled as though microvolts, so its gain matches the true microvolt channels'.
        assert list(recording.get_channel_ids()) == ["ch0", "ch1", "SpO2"]
        assert recording.get_channel_gains()[2] == recording.get_channel_gains()[0]

        recording = EDFDRecordingExtractor(file_path=path, channels_to_skip=["SpO2"])
        assert list(recording.get_channel_ids()) == ["ch0", "ch1"]

    def test_one_warning_per_unit_not_per_channel(self, tmp_path, digital_data):
        """A file whose channels share an odd unit must not emit one warning per channel."""
        path = write_edf(
            tmp_path / "many.edf",
            record_onsets=GAPPED_ONSETS,
            data=digital_data,
            channel_names=["a", "b", "c"],
            dimensions=["%", "%", "bpm"],
        )
        with pytest.warns(UserWarning) as records:
            EDFDRecordingExtractor(file_path=path)
        messages = [str(record.message) for record in records]
        assert len(messages) == 2  # one for "%", one for "bpm", not three for three channels
        assert any("'a', 'b'" in message for message in messages)

    def test_voltage_units_do_not_warn(self, tmp_path, digital_data, recwarn):
        path = write_edf(tmp_path / "v.edf", record_onsets=GAPPED_ONSETS, data=digital_data)
        EDFDRecordingExtractor(file_path=path)
        assert [record for record in recwarn if "not a voltage" in str(record.message)] == []

    @pytest.mark.parametrize("unit, expected_gain", [("uV", 1.0), ("µV", 1.0), ("mV", 1e3), ("V", 1e6), ("nV", 1e-3)])
    def test_voltage_units_scale_exactly(self, tmp_path, digital_data, unit, expected_gain):
        """Exact powers of ten — deriving these by division gave 1000.0000000000001 for millivolts."""
        path = write_edf(
            tmp_path / "unit.edf",
            record_onsets=GAPPED_ONSETS,
            data=digital_data,
            dimensions=[unit] * NUMBER_OF_CHANNELS,
        )
        recording = EDFDRecordingExtractor(file_path=path)
        # Symmetric ranges give gain = physical_range / 65536 * unit factor.
        assert recording.get_channel_gains()[0] == 2000.0 / 65536 * expected_gain

    def test_truncated_file_warns_about_the_dropped_records(self, tmp_path, digital_data):
        """Losing records to an interrupted acquisition should not be silent either."""
        path = write_edf(tmp_path / "d.edf", record_onsets=CONTIGUOUS_ONSETS, data=digital_data)
        full = Path(path).read_bytes()
        with open(path, "rb") as file:
            header = read_edf_header(file)
        Path(path).write_bytes(full[: -header.record_size_bytes])  # one whole record short

        with pytest.warns(UserWarning, match="declares 12 data records but only 11"):
            recording = EDFDRecordingExtractor(file_path=path)
        assert recording.get_num_samples(0) == 11 * SAMPLES_PER_RECORD

    def test_pre_existing_epochs_do_not_break_the_conversion(self, tmp_path, digital_data):
        """
        Reaching into nwbfile.epochs would raise once that shared table already has rows.

        Reachable from any ConverterPipe where another interface writes epochs first, or from a caller
        passing a pre-built NWBFile.
        """
        from pynwb.testing.mock.file import mock_NWBFile

        path = write_edf(tmp_path / "d.edf", record_onsets=GAPPED_ONSETS, data=digital_data)
        interface = EDFRecordingInterface(file_path=path)

        nwbfile = mock_NWBFile()
        nwbfile.add_epoch(start_time=0.0, stop_time=1.0, tags=["baseline"])
        interface.add_to_nwbfile(nwbfile=nwbfile, metadata=interface.get_metadata())

        # The caller's epoch is untouched and the runs went to their own table.
        assert len(nwbfile.epochs) == 1
        assert list(nwbfile.intervals["runs"].to_dataframe()["run_index"]) == [0, 1, 2]

    def test_split_by_offset_composes_with_the_edfd_reader(self, tmp_path, digital_data):
        """Heterogeneous offsets are still handled by the existing primitive on this path."""
        path = write_edf(
            tmp_path / "offsets.edf",
            record_onsets=GAPPED_ONSETS,
            data=digital_data,
            physical_min=[-1000.0, 0.0, -1000.0],
            physical_max=[1000.0, 1000.0, 1000.0],
        )
        interface = EDFRecordingInterface(file_path=path)
        sub_interfaces = interface.split_by_offset()

        assert len(sub_interfaces) == 2
        assert [sub.es_key for sub in sub_interfaces] == ["ElectricalSeries0", "ElectricalSeries1"]
        assert [list(sub.recording_extractor.get_channel_ids()) for sub in sub_interfaces] == [
            ["ch0", "ch2"],
            ["ch1"],
        ]
        # Each sub-interface keeps every segment of the discontinuous recording.
        assert all(sub.recording_extractor.get_num_segments() == 3 for sub in sub_interfaces)

    def test_split_by_offset_writes_annotations_only_once(self, tmp_path, digital_data):
        """All sub-interfaces share the file's annotations, but only one table may be written."""
        from neuroconv import ConverterPipe

        annotations = [(2.5, None, "Marker: 601")]
        path = write_edf(
            tmp_path / "offsets.edf",
            record_onsets=GAPPED_ONSETS,
            data=digital_data,
            annotations=annotations,
            physical_min=[-1000.0, 0.0, -1000.0],
            physical_max=[1000.0, 1000.0, 1000.0],
        )
        interface = EDFRecordingInterface(file_path=path)
        sub_interfaces = interface.split_by_offset()
        converter = ConverterPipe(data_interfaces={sub.es_key: sub for sub in sub_interfaces})

        metadata = converter.get_metadata()
        metadata["NWBFile"].update(session_start_time=datetime(2021, 9, 16, 12, 35, 13))
        nwbfile_path = tmp_path / "offsets.nwb"
        converter.run_conversion(nwbfile_path=str(nwbfile_path), metadata=metadata, overwrite=True)

        with NWBHDF5IO(str(nwbfile_path), "r") as io:
            nwbfile = io.read()
            # Two offsets crossed with three runs: the offset index comes from es_key and the run
            # index is appended by the multi-segment writer.
            assert sorted(nwbfile.acquisition) == [
                "ElectricalSeries00",
                "ElectricalSeries01",
                "ElectricalSeries02",
                "ElectricalSeries10",
                "ElectricalSeries11",
                "ElectricalSeries12",
            ]
            assert len(nwbfile.intervals["annotations"].to_dataframe()) == 1
            # And the run rows are written once, not once per sub-interface.
            assert list(nwbfile.intervals["runs"].to_dataframe()["run_index"]) == [0, 1, 2]


def _metadata(interface) -> dict:
    metadata = interface.get_metadata()
    metadata["NWBFile"].update(session_start_time=datetime(2021, 9, 16, 12, 35, 13))
    return metadata
