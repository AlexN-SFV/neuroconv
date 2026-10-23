"""
Cut the MEF 2 test fixtures from the full source files, and record what those files decode to.

Usage::

    python build_mef2_fixtures.py <Discover 13 AD3.mef> <PennEPI00082 LI07.mef> <Discover 13 annotations.xml>

The sources are public:

* Pennsieve Discover dataset 13, "Intracranial EEG Epilepsy - Study 5" (Mayo, CC0 1.0), version 1:
  ``packages/EEG/Study 5 IEEG/AD3.mef`` (MEF 2.0) and ``packages/EEG/annotations.xml``.
* Pennsieve Discover dataset 549, "PennEPI00082" (Penn CNT, Creative Commons Attribution), version 1:
  ``files/primary/sub-PennEPI00082/ses-postimplant/ieeg/sub-PennEPI00082_ses-postimplant_task-clinical_ieeg-mef/LI07.mef``
  (MEF 2.1).

Both download without an account through ``POST https://api.pennsieve.io/zipit/discover`` with the body
``{"data": {"paths": [<path>], "datasetId": <id>, "version": <version>}}``, which returns the file itself.

Each fixture keeps the chosen blocks byte for byte; only the header and the index are rewritten, by the
test module's ``write_mef2``. ``expected.json`` holds, for each fixture, the samples the full source file
decodes to over those blocks, and the runs the blocks form, grouped here by hand from the source's
discontinuity flags rather than by the reader under test.
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

from test_mef2_interface import read_mef2_file, write_mef2  # noqa: E402

from neuroconv.datainterfaces.ecephys.mef._mef2_reader import (  # noqa: E402
    read_mef2_recording,
)

# Fixture name: (source, the runs as groups of source block numbers). The first PennEPI block is the
# one-sample placeholder, which the reader drops and which belongs to no run.
FIXTURES = {
    "Discover13_AD3_excerpt": ("ad3", [list(range(0, 5)), list(range(5, 8))]),
    "Discover13_AD3_clock_step": ("ad3", [list(range(21379, 21384))]),
    "PennEPI00082_LI07_excerpt": (
        "li07",
        [[1, 2], [142494], [142495], [142496], [142497], [142498], [142499], [142500, 142501]],
    ),
}


def main(ad3_path: Path, li07_path: Path, annotations_path: Path):
    sources = {"ad3": Path(ad3_path), "li07": Path(li07_path)}
    expected = {}
    for name, (source, runs) in FIXTURES.items():
        source_path = sources[source]
        numbers = ([0] if source == "li07" else []) + [number for run in runs for number in run]
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
            # The reader's blocks skip the placeholder of a PennEPI file.
            shift = 1 if channel.dropped_placeholder_block else 0
            samples = np.concatenate([channel.read_blocks(number - shift, 1) for run in runs for number in run])
            counts = {number: struct.unpack_from("<I", blocks[number], 20)[0] for run in runs for number in run}
            expected[name] = dict(
                source=f"{source_path.name}, blocks {numbers[0]}-{numbers[-1]} ({len(numbers)} blocks)",
                number_of_samples=int(samples.size),
                sha256=hashlib.sha256(samples.astype("<i4").tobytes()).hexdigest(),
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
    (HERE / "expected.json").write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(*sys.argv[1:4])
