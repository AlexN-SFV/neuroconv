"""
``TimeIntervals`` tables shared by the recordings written into one NWB file.

Used by the readers that carry a recording's own events alongside its samples (EDF+ annotations and runs,
BrainVision markers and segments, EEGLAB events and epochs), so several recordings written into one NWB file
through a ``ConverterPipe`` share each table, with a ``source_file`` column saying which file each row came
from, rather than the second recording raising on the duplicate table name or losing its rows.
"""

import warnings
import weakref
from pathlib import Path

from pynwb import NWBFile

# For each NWB file, the (resolved source path, table name) pairs whose rows have been written to it. Every
# interface over one source file carries the same rows, whether it is a split_by_offset partition or was
# built separately (an EEG interface beside an auxiliary one with the complementary channels_to_skip), so the
# file is what is recorded, not the interface. Keyed weakly by the NWBFile itself, not by id(), which Python
# reuses once a file is discarded.
_INTERVALS_WRITTEN = weakref.WeakKeyDictionary()


def add_rows_to_intervals_table(
    nwbfile: NWBFile,
    *,
    file_path,
    name: str,
    description: str,
    columns: dict,
    rows: list[dict],
    kind: str,
    source_file_description: str,
):
    """
    Add ``rows`` read from ``file_path`` to the ``TimeIntervals`` table ``name``, creating it if the file has none.

    Each row gets a ``source_file`` column holding the name of ``file_path``, so no file's rows are lost
    when several are written into one NWB file. Every interface over one source file carries the same rows,
    so only the first of them to reach this NWB file writes them. A table of that name with other columns
    came from something else and is left alone, with a warning.

    The table named ``"epochs"`` is ``nwbfile.epochs``; every other name is an entry of ``nwbfile.intervals``.

    Which files have been written is known only for the NWB file in memory. Appending on disk to a file
    that already holds a source file's rows writes them again.

    Parameters
    ----------
    nwbfile : NWBFile
        The NWB file to add the rows to.
    file_path : str or Path
        The source file the rows were read from. Interfaces over the same file, by any path that resolves to
        it, share one set of rows.
    name : str
        Name of the table.
    description : str
        Description of the table, used only if it is created.
    columns : dict
        Column name to description, for the columns beyond ``start_time`` and ``stop_time`` that each row
        holds. ``source_file`` is added.
    rows : list of dict
        The rows, each a dict of ``start_time``, ``stop_time`` and every column in ``columns``.
    kind : str
        The format, as the warning about a foreign table names it, such as ``"EDF"``.
    source_file_description : str
        Description of the ``source_file`` column.
    """
    from pynwb.epoch import TimeIntervals

    # The source file is recognised by its resolved path, not by comparing rows: two recordings with the
    # same file name and the same layout can have identical rows, and both must be kept.
    file_path = Path(file_path)
    written = _INTERVALS_WRITTEN.setdefault(nwbfile, set())
    key = (str(file_path.resolve()), name)
    if key in written:
        return
    # Marked before writing, so a clash with a foreign table warns once per source file, not once per interface.
    written.add(key)

    source_file = file_path.name
    columns = dict(columns, source_file=source_file_description)

    table = _get_table(nwbfile=nwbfile, name=name)
    if table is None:
        table = TimeIntervals(name=name, description=description)
        for column_name, column_description in columns.items():
            table.add_column(name=column_name, description=column_description)
        _add_table(nwbfile=nwbfile, table=table)
    else:
        expected = {"start_time", "stop_time", *columns}
        if set(table.colnames) != expected:
            # Called from an interface's _add_*_to_nwbfile, itself called from add_to_nwbfile, so the warning
            # points at the caller of add_to_nwbfile.
            warnings.warn(
                f"The NWB file already has an intervals table named {name!r}, with columns "
                f"{sorted(table.colnames)} rather than {sorted(expected)}, so it holds something other than "
                f"{kind} {name}. The {name} of '{source_file}' are not written.",
                UserWarning,
                stacklevel=4,
            )
            return

    for row in rows:
        table.add_row(**row, source_file=source_file)


def _get_table(nwbfile: NWBFile, name: str):
    if name == "epochs":
        return nwbfile.epochs
    if nwbfile.intervals is not None and name in nwbfile.intervals:
        return nwbfile.intervals[name]
    return None


def _add_table(nwbfile: NWBFile, table):
    # epochs is a field of its own, written to /intervals/epochs. A table of that name added to
    # nwbfile.intervals instead is written to the same group without an error, and one of the two is lost.
    if table.name == "epochs":
        nwbfile.epochs = table
    else:
        nwbfile.add_time_intervals(table)
