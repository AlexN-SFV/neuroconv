"""The ``runs`` table the MEF 2 and MEF 3 interfaces share, defined once so either can add rows to the other's."""

from pathlib import Path

from pynwb import NWBFile

from .._intervals_tables import add_rows_to_intervals_table

RUNS_TABLE_NAME = "runs"
RUNS_TABLE_DESCRIPTION = (
    "Uninterrupted runs of samples in recordings with gaps. Each run is written as its own ElectricalSeries, "
    "positioned at the run's start time."
)
RUNS_COLUMNS = {"run_index": "0-based index of the uninterrupted run of samples within its recording."}
RUNS_SOURCE_FILE_DESCRIPTION = (
    "Name of the MEF recording the row was read from: a MEF 3 .mefd session directory, or a MEF 2 directory of "
    ".mef files, followed by the sampling rate in parentheses when the directory holds channels at several."
)


def add_runs_to_nwbfile(
    nwbfile: NWBFile,
    *,
    recording_extractor,
    run_samples: list[int],
    sampling_frequency: float,
    file_path: Path,
    kind: str,
    source_file: str | None = None,
):
    """
    Add one row per run to the ``runs`` table, each where the recording's segment for it starts.

    The start times are read from the recording, so any alignment moves the rows with the series.
    """
    rows = []
    for run_index, number_of_samples in enumerate(run_samples):
        start_time = float(recording_extractor.get_start_time(segment_index=run_index))
        rows.append(
            dict(
                start_time=start_time,
                stop_time=start_time + number_of_samples / sampling_frequency,
                run_index=run_index,
            )
        )
    add_rows_to_intervals_table(
        nwbfile=nwbfile,
        file_path=file_path,
        name=RUNS_TABLE_NAME,
        description=RUNS_TABLE_DESCRIPTION,
        columns=RUNS_COLUMNS,
        rows=rows,
        kind=kind,
        source_file_description=RUNS_SOURCE_FILE_DESCRIPTION,
        source_file=source_file,
    )
