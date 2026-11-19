# MEF 2 test fixtures

Built by `build_mef2_fixtures.py`, which says how to download the sources of the excerpts and rebuild
everything.

- `Discover13_AD3_excerpt/`, `Discover13_AD3_clock_step/` and `Discover13_annotations_excerpt.xml`: excerpts of
  "Intracranial EEG Epilepsy - Study 5", Mayo Clinic, Pennsieve Discover dataset 13, version 1, released under
  CC0 1.0. MEF 2.0. Each `.mef` file keeps its source's blocks byte for byte; only its header and block index
  are rewritten for the shorter file.
- `mef21_dropout_runs/`: a generated MEF 2.1 file of one channel, with a placeholder first block and 8 runs
  split by dropouts. It holds no recorded data.
