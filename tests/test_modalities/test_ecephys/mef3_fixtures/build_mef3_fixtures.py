"""
Build the MEF 3 fixtures in this directory, and the values pymef reads from them, in ``expected.json``.

Run it with pymef installed (pymef 1.4.8 ships wheels for x86_64 only), from anywhere:

    python build_mef3_fixtures.py <ds004696 sub-06 .mefd directory>

pymef is the oracle and the writer here, and nothing else: the tests compare neuroconv's reader with what
pymef read at build time, so they need neither pymef nor the network. The sessions are written with
random UUIDs, so rebuilding them changes their bytes but not what they hold.

* ``gapped.mefd``: three channels at 1000 Hz, in two segments with a 2 s gap between them, the first
  segment written in two parts (pymef 1.4.8 flags neither the appended part nor anything but each segment's
  first block, whatever ``discontinuity_flag`` says; the tests patch flags in copies where they need them). Records
  at session, channel and segment level; a recording time offset; a GMT offset; a negative units conversion
  factor and a channel in millivolts.
* ``lossy.mefd``: one channel written with RED's lossy compression, which scales and detrends its blocks.
* ``encrypted.mefd``: one channel written with passwords, so its metadata is encrypted.
* ``ds004696_sub-06_truncated.mefd``: channels C3 and EKG of ds004696 sub-06 (CC0, OpenNeuro), cut to their
  first six blocks. Block 3 holds 10242 samples though the next block starts 10240 samples later: the clock
  drift that makes timestamps alone the wrong test for a gap.
"""

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from pymef.mef_file import pymef3_file
from pymef.mef_session import MefSession

HERE = Path(__file__).resolve().parent
START_TIME = 946684800000000  # 2000-01-01T00:00:00 UTC, in µUTC
RECORDING_TIME_OFFSET = START_TIME - 1_000_000
SAMPLING_FREQUENCY = 1000
SAMPLES_PER_BLOCK = 250


def digest(samples) -> str:
    return hashlib.sha256(np.asarray(samples, dtype="<i4").tobytes()).hexdigest()


def section_2(**fields) -> dict:
    defaults = dict(
        channel_description=b"",
        session_description=b"",
        recording_duration=0,
        reference_description=b"",
        acquisition_channel_number=0,
        sampling_frequency=SAMPLING_FREQUENCY,
        notch_filter_frequency_setting=60.0,
        low_frequency_filter_setting=0.5,
        high_frequency_filter_setting=300.0,
        AC_line_frequency=60.0,
        units_conversion_factor=1.0,
        units_description=b"microvolts",
        maximum_native_sample_value=0.0,
        minimum_native_sample_value=0.0,
        start_sample=0,
        number_of_blocks=0,
        maximum_block_bytes=0,
        maximum_block_samples=0,
        maximum_difference_bytes=0,
        block_interval=0,
        number_of_discontinuities=0,
        maximum_contiguous_blocks=0,
        maximum_contiguous_block_bytes=0,
        maximum_contiguous_samples=0,
        number_of_samples=0,
    )
    return {**defaults, **fields}


SECTION_3 = dict(
    recording_time_offset=RECORDING_TIME_OFFSET,
    DST_start_time=0,
    DST_end_time=0,
    GMT_offset=3600,
    subject_name_1=b"",
    subject_name_2=b"",
    subject_ID=b"",
    recording_location=b"",
)


def fresh(path: Path) -> MefSession:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir()
    return MefSession(str(path) + "/", None, read_metadata=False)


def build_gapped(path: Path) -> dict:
    rng = np.random.default_rng(seed=3)
    session = fresh(path)
    # Name order (Cz, EKG, Fp1) differs from acquisition order (Cz, Fp1, EKG).
    channels = {
        "Fp1": dict(acquisition_channel_number=2, units_conversion_factor=-0.25, reference_description=b"Cz"),
        "Cz": dict(acquisition_channel_number=1, units_conversion_factor=0.5),
        "EKG": dict(acquisition_channel_number=3, units_conversion_factor=0.001, units_description=b"millivolts"),
    }
    first_part, appended, second_segment = 1500, 500, 1000
    gap = 2_000_000
    segment_1_start = START_TIME + (first_part + appended) * 1000 + gap
    data = {}
    for name, fields in channels.items():
        # Small steps with occasional jumps, so blocks hold both ordinary differences and keysamples.
        steps = rng.integers(-40, 40, first_part + appended + second_segment)
        steps[rng.integers(0, steps.size, 20)] = rng.integers(-300000, 300000, 20)
        samples = np.cumsum(steps).astype(np.int32)
        data[name] = samples
        session.write_mef_ts_segment_metadata(
            name, 0, None, None, START_TIME, START_TIME + (first_part + appended) * 1000, section_2(**fields), SECTION_3
        )
        session.write_mef_ts_segment_data(name, 0, None, None, SAMPLES_PER_BLOCK, samples[:first_part])
        # Appended exactly where the first part ends. pymef ignores the flag here; it is passed to show the attempt.
        session.append_mef_ts_segment_data(
            name,
            0,
            None,
            None,
            START_TIME + first_part * 1000,
            START_TIME + (first_part + appended) * 1000,
            SAMPLES_PER_BLOCK,
            samples[first_part : first_part + appended],
            discontinuity_flag=True,
        )
        session.write_mef_ts_segment_metadata(
            name,
            1,
            None,
            None,
            segment_1_start,
            segment_1_start + second_segment * 1000,
            section_2(start_sample=first_part + appended, **fields),
            SECTION_3,
        )
        session.write_mef_ts_segment_data(name, 1, None, None, SAMPLES_PER_BLOCK, samples[first_part + appended :])

    end = segment_1_start + second_segment * 1000
    session.write_mef_records(
        None,
        None,
        START_TIME,
        end,
        RECORDING_TIME_OFFSET,
        [
            dict(type="Note", time=START_TIME + 100_000, text="session note"),
            dict(type="EDFA", time=START_TIME + 2_500_000, duration=1_500_000, text="artifact"),
        ],
    )
    session.write_mef_records(
        None,
        None,
        START_TIME,
        end,
        RECORDING_TIME_OFFSET,
        [
            dict(
                type="Seiz",
                time=START_TIME + 300_000,
                earliest_onset=START_TIME + 300_000,
                latest_offset=START_TIME + 800_000,
                duration=500_000,
                number_of_channels=1,
                onset_code=1,
                marker_name_1="m1",
                marker_name_2="m2",
                annotation="focal onset",
                channels=[dict(name="Cz", onset=START_TIME + 300_000, offset=START_TIME + 800_000)],
            )
        ],
        channel="Cz",
    )
    session.write_mef_records(
        None,
        None,
        START_TIME,
        end,
        RECORDING_TIME_OFFSET,
        [
            dict(
                type="ESti",
                time=START_TIME + 1_234_000,
                amplitude=6.0,
                frequency=1.0,
                pulse_width=200,
                ampunit_code=1,
                mode_code=1,
                waveform="biphasic",
                anode="Cz",
                catode="Fp1",
            ),
            dict(
                type="CSti",
                time=START_TIME + 1_500_000,
                task_type="naming",
                stimulus_duration=250_000,
                stimulus_type="picture",
                patient_response="correct",
            ),
        ],
        channel="Cz",
        segment_n=0,
    )
    return data


def build_lossy(path: Path) -> dict:
    session = fresh(path)
    abscissa = np.arange(2000)
    samples = (3000 * np.sin(abscissa / 25) + 0.9 * abscissa + np.random.default_rng(4).normal(0, 40, 2000)).astype(
        np.int32
    )
    session.write_mef_ts_segment_metadata(
        "ch", 0, None, None, START_TIME, START_TIME + 2_000_000, section_2(), SECTION_3
    )
    segment = f"{path}/ch.timd/ch-000000.segd/"
    # MefSession.write_mef_ts_segment_data always passes lossy_flag=0; the C function takes it.
    pymef3_file.write_mef_ts_data_and_indices(segment, None, None, 1000, samples, 1)
    return dict(ch=samples)


def build_encrypted(path: Path):
    session = fresh(path)
    samples = np.arange(500, dtype=np.int32)
    session.write_mef_ts_segment_metadata(
        "ch", 0, "chair", "table", START_TIME, START_TIME + 500_000, section_2(), SECTION_3
    )
    session.write_mef_ts_segment_data("ch", 0, "chair", "table", SAMPLES_PER_BLOCK, samples)


def build_truncated(source: Path, path: Path, channels=("C3", "EKG"), number_of_blocks=6):
    if path.exists():
        shutil.rmtree(path)
    for channel in channels:
        source_segment = source / f"{channel}.timd" / f"{channel}-000000.segd"
        segment = path / f"{channel}.timd" / f"{channel}-000000.segd"
        segment.mkdir(parents=True)
        stem = f"{channel}-000000"
        index = bytearray((source_segment / f"{stem}.tidx").read_bytes()[: 1024 + 56 * number_of_blocks])
        entries = np.frombuffer(bytes(index[1024:]), dtype=np.uint8).reshape(-1, 56)
        offsets = entries[:, :8].copy().view("<i8").ravel()
        block_bytes = entries[:, 28:32].copy().view("<u4").ravel()
        samples = int(entries[:, 24:28].copy().view("<u4").sum())
        index[32:40] = np.int64(number_of_blocks).tobytes()  # UNIVERSAL_HEADER_NUMBER_OF_ENTRIES_OFFSET
        (segment / f"{stem}.tidx").write_bytes(bytes(index))
        data = bytearray((source_segment / f"{stem}.tdat").read_bytes()[: int(offsets[-1] + block_bytes[-1])])
        data[32:40] = np.int64(number_of_blocks).tobytes()
        (segment / f"{stem}.tdat").write_bytes(bytes(data))
        metadata = bytearray((source_segment / f"{stem}.tmet").read_bytes())
        metadata[8920:8928] = np.int64(samples).tobytes()  # TIME_SERIES_METADATA_NUMBER_OF_SAMPLES_OFFSET
        metadata[8928:8936] = np.int64(number_of_blocks).tobytes()  # TIME_SERIES_METADATA_NUMBER_OF_BLOCKS_OFFSET
        (segment / f"{stem}.tmet").write_bytes(bytes(metadata))


def read_with_pymef(path: Path) -> dict:
    session = MefSession(str(path), None)
    channels = {}
    for name, channel in session.session_md["time_series_channels"].items():
        number_of_samples = sum(
            int(segment["section_2"]["number_of_samples"][0]) for segment in channel["segments"].values()
        )
        samples = np.asarray(session.read_ts_channels_sample(name, [0, number_of_samples])).astype(np.int64)
        toc = session.get_channel_toc(name)
        channels[name] = dict(
            number_of_samples=number_of_samples,
            sha256=digest(samples),
            first_samples=samples[:5].tolist(),
            last_samples=samples[-5:].tolist(),
            # pymef's table of contents, one column per block: its discontinuity flag, number of samples and
            # start time with the recording time offset removed.
            block_discontinuity=[bool(flag) for flag in toc[0]],
            block_samples=[int(count) for count in toc[1]],
            block_start_times=[int(time) for time in toc[3]],
            units_conversion_factor=float(channel["section_2"]["units_conversion_factor"][0]),
        )

    def read_records(**level) -> list:
        # pymef raises KeyError for a level that holds no records.
        try:
            return session.read_records(**level)
        except KeyError:
            return []

    records = []
    levels = [dict()]
    for name, channel in session.session_md["time_series_channels"].items():
        levels.append(dict(channel=name))
        levels += [dict(channel=name, segment_n=segment) for segment in range(len(channel["segments"]))]
    for level in levels:
        for record in read_records(**level):
            fields = {key: value for key, value in record.items() if key in ("text", "duration", "annotation")}
            records.append(
                dict(time=int(record["time"]), type=record["type"], channel=level.get("channel", ""), **fields)
            )
    return dict(channels=channels, records=records)


def main():
    ds004696 = Path(sys.argv[1])
    expected = {}

    written = build_gapped(HERE / "gapped.mefd")
    expected["gapped.mefd"] = read_with_pymef(HERE / "gapped.mefd")
    for name, samples in written.items():
        assert expected["gapped.mefd"]["channels"][name]["sha256"] == digest(samples), name

    written = build_lossy(HERE / "lossy.mefd")
    expected["lossy.mefd"] = read_with_pymef(HERE / "lossy.mefd")
    expected["lossy.mefd"]["original_sha256"] = digest(written["ch"])

    build_encrypted(HERE / "encrypted.mefd")

    build_truncated(ds004696, HERE / "ds004696_sub-06_truncated.mefd")
    expected["ds004696_sub-06_truncated.mefd"] = read_with_pymef(HERE / "ds004696_sub-06_truncated.mefd")

    (HERE / "expected.json").write_text(json.dumps(expected, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
