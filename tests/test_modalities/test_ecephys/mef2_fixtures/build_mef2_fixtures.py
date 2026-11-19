"""
Build the MEF 2 test fixtures, and record what they hold in ``expected.json``.

Usage::

    python build_mef2_fixtures.py synthetic
    python build_mef2_fixtures.py <Discover 13 AD3.mef> <Discover 13 annotations.xml>

Each form rebuilds only its own fixtures and updates only their entries in ``expected.json``.

``synthetic`` generates ``mef21_dropout_runs/``, a MEF 2.1 file laid out as IEEG.org's MEF 2.1 writer lays out
a recording: one channel at 1024 Hz whose first block is a placeholder of one sample of value 0, stamped with
the header start, 2.5 s before the recording proper; then 8 runs split by dropouts of whole samples, each run's
first block flagged and listed in the discontinuity index (9 entries with the placeholder's and the first real
block's). It has a negative voltage conversion factor and a GMT offset. The samples are a fixed signal whose
differences mostly fit in a byte but not always, so blocks hold keysamples after their first sample, and with a
quiet stretch whose symbol counts exceed 255 and are scaled. The blocks are coded by the test module's
``encode_red_block``, which a test checks rewrites the blocks of the real MEF 2.0 excerpts byte for byte.
``expected.json`` holds the samples generated, not anything a reader decoded.

The second form cuts excerpts of a public recording: Pennsieve Discover dataset 13, "Intracranial EEG Epilepsy -
Study 5" (Mayo, CC0 1.0), version 1: ``packages/EEG/Study 5 IEEG/AD3.mef`` (MEF 2.0) and
``packages/EEG/annotations.xml``. They download without an account through
``POST https://api.pennsieve.io/zipit/discover`` with the body
``{"data": {"paths": [<path>], "datasetId": 13, "version": 1}}``, which returns the file itself.

Each excerpt keeps the chosen blocks byte for byte; only the header and the index are rewritten, by the test
module's ``write_mef2``. ``expected.json`` holds, for each excerpt, the samples the full source file decodes to
over those blocks, and the runs the blocks form, grouped here by hand from the source's discontinuity flags
rather than by the reader under test.
"""

import hashlib
import json
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

from test_mef2_interface import (  # noqa: E402
    BLOCK_HEADER_BYTES,
    HEADER_BYTES,
    encode_red_block,
    patch_header,
    read_mef2_file,
    write_mef2,
)

from neuroconv.datainterfaces.ecephys.mef._mef2_reader import (  # noqa: E402
    read_mef2_recording,
)

# Excerpt name: the runs as groups of source block numbers.
EXCERPTS = {
    "Discover13_AD3_excerpt": [list(range(0, 5)), list(range(5, 8))],
    "Discover13_AD3_clock_step": [list(range(21379, 21384))],
}

SYNTHETIC = "mef21_dropout_runs"
SYNTHETIC_CHANNEL = "CH01"
SYNTHETIC_RATE = 1024.0
SYNTHETIC_HEADER_START = 946728000000000  # 2000-01-01T12:00:00 UTC, in µUTC
PLACEHOLDER_LEAD = 2_500_000  # µs from the placeholder block to the recording proper
# The sample counts of each run's blocks, and the dropout after each run but the last, in whole samples.
RUN_BLOCKS = [[2500, 2500], [1000], [1400], [90], [30], [900], [180], [2500, 2500]]
DROPOUTS = [256, 16, 480, 48, 560, 64, 245_760]
VOLTAGE_CONVERSION_FACTOR = -0.005
GMT_OFFSET = -5.0


def digest(samples) -> str:
    return hashlib.sha256(np.asarray(samples, dtype="<i4").tobytes()).hexdigest()


def synthetic_samples(number_of_samples: int) -> np.ndarray:
    """Two sinusoids and noise, with a quiet stretch and a few steps, as int32 counts."""
    rng = np.random.default_rng(2)
    time = np.arange(number_of_samples) / SYNTHETIC_RATE
    signal = 2000 * np.sin(2 * np.pi * 7 * time) + 600 * np.sin(2 * np.pi * 31 * time + 1) - 300
    signal += rng.normal(0, 15, number_of_samples)
    quiet = slice(2600, 3600)
    signal[quiet] = -300 + rng.normal(0, 2, quiet.stop - quiet.start)
    for position in (700, 5400, 9100):
        signal[position:] += 1500 * np.sign(rng.normal())
    return np.rint(signal).astype(np.int32)


def build_synthetic(expected: dict):
    number_of_samples = sum(sum(run) for run in RUN_BLOCKS)
    samples = synthetic_samples(number_of_samples)

    blocks = [encode_red_block([0], SYNTHETIC_HEADER_START, discontinuity=True)]
    time = float(SYNTHETIC_HEADER_START + PLACEHOLDER_LEAD)
    position = 0
    run_start_times = []
    for run, counts in enumerate(RUN_BLOCKS):
        run_start_times.append(int(np.rint(time)))
        for number, count in enumerate(counts):
            block_samples = samples[position : position + count]
            blocks.append(encode_red_block(block_samples, int(np.rint(time)), discontinuity=number == 0))
            position += count
            time += count * 1e6 / SYNTHETIC_RATE
        if run < len(DROPOUTS):
            time += DROPOUTS[run] * 1e6 / SYNTHETIC_RATE

    # What the fixture is for: keysamples after a block's first sample, and symbol counts that were scaled.
    coded_bytes = [struct.unpack_from("<I", block, 16)[0] for block in blocks[1:]]
    block_samples = [struct.unpack_from("<I", block, 20)[0] for block in blocks[1:]]
    assert any(coded > count + 2 for coded, count in zip(coded_bytes, block_samples))
    assert any(max(block[31:BLOCK_HEADER_BYTES]) == 255 for block in blocks[1:])

    header = bytearray(HEADER_BYTES)
    header[128:139] = b"128-bit AES"
    header[163:166] = bytes([1, 2, 1])  # little-endian, MEF 2.1
    struct.pack_into("<H", header, 166, HEADER_BYTES)
    header[168:176] = bytes(range(1, 9))  # session unique ID
    header[948:956] = bytes(range(9, 17))  # file unique ID
    struct.pack_into(
        "<IQQii",
        header,
        788,
        max(len(block) for block in blocks),
        max(max(run) for run in RUN_BLOCKS),
        int(np.rint(2500 * 1e6 / SYNTHETIC_RATE)),
        int(samples.max()),
        int(samples.min()),
    )
    struct.pack_into("<H", header, 832, BLOCK_HEADER_BYTES)
    header = patch_header(
        bytes(header),
        channel_name=SYNTHETIC_CHANNEL,
        sampling_frequency=SYNTHETIC_RATE,
        voltage_conversion_factor=VOLTAGE_CONVERSION_FACTOR,
        physical_channel_number=1,
        gmt_offset=GMT_OFFSET,
        compression_algorithm="Range Encoded Differences (RED)",
    )
    folder = HERE / SYNTHETIC
    for stale in folder.glob("*.mef"):
        stale.unlink()
    write_mef2(folder / f"{SYNTHETIC_CHANNEL}.mef", header, blocks)

    expected[SYNTHETIC] = dict(
        source="synthetic, generated by build_mef2_fixtures.py",
        number_of_samples=int(samples.size),
        sha256=digest(samples),
        run_samples=[sum(run) for run in RUN_BLOCKS],
        run_start_times=run_start_times,
    )


def build_excerpts(expected: dict, ad3_path: Path, annotations_path: Path):
    source_path = Path(ad3_path)
    for name, runs in EXCERPTS.items():
        numbers = [number for run in runs for number in run]
        header, selected = read_mef2_file(source_path, numbers)
        blocks = dict(zip(numbers, selected))

        # The flags of the copied blocks must say what the source's discontinuity index says, since the
        # fixture's own index is rebuilt from the flags.
        disc_offset, disc_count = struct.unpack_from("<QQ", header, 840) if header[165] >= 1 else (0, 0)
        if disc_count:
            raw = source_path.read_bytes()[disc_offset : disc_offset + 8 * disc_count]
            indexed = set((np.frombuffer(raw, dtype="<u8") - 1).tolist())
            assert all(bool(blocks[number][30] & 1) == (number in indexed) for number in numbers), name

        folder = HERE / name
        folder.mkdir(exist_ok=True)
        write_mef2(folder / source_path.name, header, [blocks[number] for number in numbers])

        # What the full source file decodes to over the same blocks.
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / source_path.name).symlink_to(source_path.resolve())
            full = read_mef2_recording(directory, validate_crc=True)
            (channel,) = full.channels
            # The reader's blocks skip a placeholder first block, where a file has one.
            shift = 1 if channel.dropped_placeholder_block else 0
            samples = np.concatenate([channel.read_blocks(number - shift, 1) for run in runs for number in run])
            counts = {number: struct.unpack_from("<I", blocks[number], 20)[0] for run in runs for number in run}
            expected[name] = dict(
                source=f"{source_path.name}, blocks {numbers[0]}-{numbers[-1]} ({len(numbers)} blocks)",
                number_of_samples=int(samples.size),
                sha256=digest(samples),
                run_samples=[sum(counts[number] for number in run) for run in runs],
                run_start_times=[struct.unpack_from("<Q", blocks[run[0]], 8)[0] for run in runs],
            )

    # The first annotation of the dataset's annotations file, verbatim.
    text = Path(annotations_path).read_text(encoding="utf-8")
    first = text[text.index("<annotation\n") : text.index("</annotation>") + len("</annotation>")]
    (HERE / "Discover13_annotations_excerpt.xml").write_text(
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<annotations>\n  ' + first + "\n</annotations>\n",
        encoding="utf-8",
    )


def main(arguments: list[str]):
    expected_path = HERE / "expected.json"
    expected = json.loads(expected_path.read_text()) if expected_path.exists() else {}
    if arguments == ["synthetic"]:
        build_synthetic(expected)
    elif len(arguments) == 2:
        build_excerpts(expected, *arguments)
    else:
        raise SystemExit(__doc__)
    expected_path.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:])
