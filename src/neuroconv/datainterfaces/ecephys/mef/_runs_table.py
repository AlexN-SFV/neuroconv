"""The ``runs`` table the MEF 2 and MEF 3 interfaces share, defined once so either can add rows to the other's."""

from pathlib import Path

from pynwb import NWBFile

from .._intervals_tables import add_rows_to_intervals_table

RUNS_TABLE_NAME = "runs"
RUNS_TABLE_DESCRIPTION = (
    "Uninterrupted runs of samples in recordings with gaps. Each run is written as its own ElectricalSeries, "
    "positioned at the run's start time, or, for a recording with many runs, as a stretch of one ElectricalSeries "
    "whose timestamps hold the gaps. A recording, or a channel group of one, that has a single run has no rows: "
    "its series' starting time places it."
)
RUNS_COLUMNS = {"run_index": "0-based index of the uninterrupted run of samples within its recording."}
RUNS_SOURCE_FILE_DESCRIPTION = (
    "Name of the MEF recording the row was read from: a MEF 3 .mefd session directory, or a MEF 2 directory of "
    ".mef files, followed by the sampling rate in parentheses when the directory holds channels at several, "
    "and by the channel group's number when channels at one rate were recorded over different stretches of time."
)


def add_runs_to_nwbfile(
    nwbfile: NWBFile,
    *,
    run_start_times,
    run_samples: list[int],
    sampling_frequency: float,
    file_path: Path,
    kind: str,
    source_file: str | None = None,
):
    """
    Add one row per run to the ``runs`` table, starting at ``run_start_times``.

    The caller reads the start times from its recording, so any alignment moves the rows with the series.
    """
    rows = []
    for run_index, (number_of_samples, start_time) in enumerate(zip(run_samples, run_start_times)):
        start_time = float(start_time)
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
