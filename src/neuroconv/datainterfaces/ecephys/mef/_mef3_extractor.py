"""
SpikeInterface recording extractor for MEF 3.0 sessions.

The parsing lives in :mod:`._mef3_reader` and the decoding in :mod:`._red`, which depend only on NumPy and,
optionally, numba. This module is where SpikeInterface enters, so it must only be imported lazily, as
:meth:`~..mef3datainterface.MEF3RecordingInterface.get_extractor_class` does.

The classes are defined at module scope because SpikeInterface records an extractor's class by its
import path in ``to_dict()``; classes built inside a function cannot be pickled or reloaded.
"""

import os
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from pydantic import DirectoryPath
from spikeinterface.core import BaseRecording, BaseRecordingSegment

from ._mef3_crypto import MEF3Passwords
from ._mef3_reader import MEF3Channel, MEF3Session, read_mef3_session
from ._red import NUMBA_AVAILABLE
from .._channel_ids import _stacklevel_outside_neuroconv
from .._voltage_units import unit_to_microvolts


class MEF3RecordingSegment(BaseRecordingSegment):
    """
    Runs of a session, decoded lazily, block by block, from each channel's data files.

    Usually the one uninterrupted run ``run_index``, placed at ``t_start``. With ``time_vector`` it is
    ``number_of_runs`` consecutive runs from ``run_index``, back to back, and the time vector gives each
    sample's time, gaps included.
    """

    def __init__(
        self,
        *,
        channels: list[MEF3Channel],
        run_index: int,
        number_of_samples: int,
        sampling_frequency: float,
        t_start: float | None = None,
        time_vector: np.ndarray | None = None,
        number_of_runs: int = 1,
    ):
        # The rate stays beside a time vector, as spikeinterface's own set_times leaves it: a frame slice of the
        # segment (a stub) rebuilds it from its times kwargs without the vector, and needs the rate then.
        super().__init__(sampling_frequency=sampling_frequency, t_start=t_start, time_vector=time_vector)
        self._channels = channels
        self._number_of_samples = number_of_samples
        # Per channel: the segment's first block and the sample each of its blocks ends at, within the segment.
        # A channel's runs partition its blocks in order, so consecutive runs are consecutive blocks.
        self._first_blocks = []
        self._block_ends = []
        for channel in channels:
            first_run, last_run = channel.runs[run_index], channel.runs[run_index + number_of_runs - 1]
            first_block = first_run.first_block
            end_block = last_run.first_block + last_run.number_of_blocks
            counts = channel.blocks["number_of_samples"][first_block:end_block]
            self._first_blocks.append(first_block)
            self._block_ends.append(np.cumsum(counts))

    def get_num_samples(self) -> int:
        return int(self._number_of_samples)

    def _read_channel(self, position: int, start_frame: int, end_frame: int) -> np.ndarray:
        """Samples ``start_frame`` to ``end_frame`` of one channel, decoding only the blocks they fall in."""
        block_ends = self._block_ends[position]
        first = int(np.searchsorted(block_ends, start_frame, side="right"))
        last = int(np.searchsorted(block_ends, end_frame - 1, side="right"))
        samples = self._channels[position].read_blocks(self._first_blocks[position] + first, last - first + 1)
        first_sample_of_blocks = int(block_ends[first]) - int(
            self._channels[position].blocks["number_of_samples"][self._first_blocks[position] + first]
        )
        offset = start_frame - first_sample_of_blocks
        return samples[offset : offset + (end_frame - start_frame)]

    def get_traces(self, start_frame=None, end_frame=None, channel_indices=None) -> np.ndarray:
        start_frame = 0 if start_frame is None else max(int(start_frame), 0)
        end_frame = self._number_of_samples if end_frame is None else min(int(end_frame), self._number_of_samples)
        positions = np.arange(len(self._channels))
        positions = np.atleast_1d(positions if channel_indices is None else positions[channel_indices])
        traces = np.empty((max(end_frame - start_frame, 0), len(positions)), dtype=np.int32)
        if end_frame <= start_frame:
            return traces

        # The compiled decoder releases the GIL, so channels are decoded on threads. The NumPy fallback holds
        # it, and gains nothing from them.
        if NUMBA_AVAILABLE and len(positions) > 1:
            with ThreadPoolExecutor(max_workers=min(len(positions), os.cpu_count() or 1)) as executor:
                columns = executor.map(lambda position: self._read_channel(position, start_frame, end_frame), positions)
                for column, samples in enumerate(columns):
                    traces[:, column] = samples
        else:
            for column, position in enumerate(positions):
                traces[:, column] = self._read_channel(position, start_frame, end_frame)
        return traces


def runs_ascend(runs, sampling_frequency: float) -> bool:
    """
    Whether every run starts after the previous run's last sample, so the runs back to back have increasing times.

    A run can start before the previous one ended: a flagged block that overlaps the one before it, or an unflagged
    jump back of more than a second, starts a new run.
    """
    starts = np.array([run.start_time for run in runs], dtype=np.int64)
    last_samples = starts[:-1] + (np.array([run.number_of_samples for run in runs[:-1]]) - 1) * (
        1e6 / sampling_frequency
    )
    return bool(np.all(starts[1:] > last_samples))


def _runs_time_vector(runs, sampling_frequency: float, session_start_time: int) -> np.ndarray:
    """Each sample's time in seconds from the session's start: every run at the nominal rate from its own start."""
    counts = np.array([run.number_of_samples for run in runs], dtype=np.int64)
    run_starts = (np.array([run.start_time for run in runs], dtype=np.int64) - session_start_time) / 1e6
    first_samples = np.concatenate([[0], np.cumsum(counts)[:-1]])
    # The sample's index in the whole vector over the rate, plus its run's start less its run's first index over the
    # rate: one float64 array the size of the result besides it, where index arithmetic would take three.
    times = np.arange(int(counts.sum()), dtype=np.float64)
    times /= sampling_frequency
    times += np.repeat(run_starts - first_samples / sampling_frequency, counts)
    return times


def _gains_to_microvolts(session: MEF3Session) -> np.ndarray:
    """Each channel's units conversion factor, in microvolts per count, with the sign it is stored with."""
    gains = np.empty(len(session.channels), dtype="float64")
    not_voltage, no_factor = {}, []
    for index, channel in enumerate(session.channels):
        factor = unit_to_microvolts(channel.units_description)
        if factor is None:
            # Still read, as microvolts with a warning, since dropping the channel is the caller's decision.
            not_voltage.setdefault(channel.units_description, []).append(channel.name)
            factor = 1.0
        conversion = channel.units_conversion_factor
        # 0.0 is TIME_SERIES_METADATA_UNITS_CONVERSION_FACTOR_NO_ENTRY.
        if conversion == 0.0 or not np.isfinite(conversion):
            no_factor.append(channel.name)
            conversion = 1.0
        gains[index] = conversion * factor

    def shown(names: list[str]) -> str:
        text = ", ".join(repr(name) for name in names[:5])
        return text + (f", ... ({len(names)} channels)" if len(names) > 5 else "")

    for unit, names in sorted(not_voltage.items()):
        declared = f"the unit {unit!r}, which is not a voltage" if unit else "no unit"
        warnings.warn(
            f"The MEF 3 channels {shown(names)} of '{session.path.name}' declare {declared}. Their values are "
            "being written to an NWB ElectricalSeries as though they were microvolts, so they will be wrong by "
            "whatever factor separates their unit from microvolts. Drop them with channels_to_skip if they are "
            "not neural data.",
            UserWarning,
            stacklevel=_stacklevel_outside_neuroconv(),
        )
    if no_factor:
        warnings.warn(
            f"The MEF 3 channels {shown(no_factor)} of '{session.path.name}' give no units conversion factor, so "
            "their samples are written as though one count were one microvolt.",
            UserWarning,
            stacklevel=_stacklevel_outside_neuroconv(),
        )
    return gains


def _warn_about_lossy_blocks(session: MEF3Session):
    if not session.lossy_blocks:
        return
    total = sum(session.lossy_blocks.values())
    shown = ", ".join(f"{name!r} ({count})" for name, count in list(session.lossy_blocks.items())[:5])
    if len(session.lossy_blocks) > 5:
        shown += f", ... ({len(session.lossy_blocks)} channels)"
    scaled = (
        f" The largest scale factor is {session.largest_scale_factor:g}: the values of that block come in steps "
        f"of about {session.largest_scale_factor:g} counts."
        if session.largest_scale_factor > 1.0
        else " None is scaled, only detrended, which RED undoes exactly up to rounding."
    )
    warnings.warn(
        f"{total} blocks of the MEF 3 session '{session.path.name}' were stored with RED's lossy options, scaled "
        f"by a factor or with a fitted line subtracted, in channels {shown}.{scaled} They are decoded as meflib "
        "decodes them, so the data match what any MEF reader returns, but where a block was scaled the precision "
        "lost when the file was written cannot be recovered.",
        UserWarning,
        stacklevel=_stacklevel_outside_neuroconv(),
    )


class MEF3RecordingExtractor(BaseRecording):
    """
    Lazy SpikeInterface recording for a MEF 3.0 session.

    Each uninterrupted run of blocks becomes one segment, placed on the session timeline at its first
    block's start time, measured from the session's start. A session recorded without interruption,
    however many MEF segments it is split into, is a single segment. With ``concatenate_runs`` the runs are
    instead one segment with a time vector: each run's samples at the nominal rate from its own start time,
    so the gaps between runs are in the times rather than between segments.

    A recording has one rate and one set of segments, so it holds one of the session's channel groups (see
    :class:`._mef3_reader.MEF3ChannelGroup`). A session with several is refused here: keep one group with
    ``channels_to_skip``, or convert it with ``MEF3RecordingInterface``, which writes each group as its own
    series.

    Parameters
    ----------
    folder_path : DirectoryPath
        The session's ``.mefd`` directory.
    channels_to_skip : list of str, optional
        Names of channels to leave out.
    password : str, optional
        The session's password, of either level, when its metadata is encrypted.
    level_1_password, level_2_password : str, optional
        The session's passwords by level, instead of ``password``.
    concatenate_runs : bool, default: False
        If True and the session has more than one run, read them as one segment with a time vector instead of
        one segment per run. A recording with thousands of short runs (a recorder that drops a few samples
        every so often) is then written as one series with timestamps, rather than as thousands of series.
        Refused when a run starts before the previous one ends, since the times would go back.

    The passwords are only needed while the session is read here, and are left out of the extractor's kwargs,
    so ``to_dict()`` and anything that serialises the extractor never carry them. An encrypted session's
    extractor therefore cannot be rebuilt from ``to_dict()``.

    ``_session`` is for ``MEF3RecordingInterface``: a session it has already read, narrowed to one channel group
    with :meth:`._mef3_reader.MEF3Session.select_group`, so the session is read once for all its groups. The
    ``channels_to_skip`` given with it must leave exactly that group, so ``to_dict()`` rebuilds the same recording.
    """

    extractor_name = "MEF3Recording"
    mode = "folder"
    name = "mef3"

    def __init__(
        self,
        folder_path: DirectoryPath,
        channels_to_skip: list | None = None,
        password: str | None = None,
        level_1_password: str | None = None,
        level_2_password: str | None = None,
        concatenate_runs: bool = False,
        *,
        _session: MEF3Session | None = None,
    ):
        if _session is None:
            passwords = MEF3Passwords.from_arguments(password, level_1_password, level_2_password)
            session = read_mef3_session(folder_path, channels_to_skip=channels_to_skip, passwords=passwords)
        else:
            session = _session
        # Raises, describing the groups, when the session has several.
        session._only_group()
        _warn_about_lossy_blocks(session)
        channel_names = [channel.name for channel in session.channels]

        super().__init__(
            sampling_frequency=session.sampling_frequency, channel_ids=channel_names, dtype=np.dtype("int32")
        )
        runs = session.runs
        if concatenate_runs and len(runs) > 1 and not runs_ascend(runs, session.sampling_frequency):
            raise ValueError(
                f"The runs of the MEF 3 session '{session.path.name}' cannot be concatenated: a run starts before "
                "the previous one ends, so their samples back to back would go back in time. Read them as one "
                "segment per run (concatenate_runs=False)."
            )
        if concatenate_runs and len(runs) > 1:
            self.add_recording_segment(
                MEF3RecordingSegment(
                    channels=session.channels,
                    run_index=0,
                    number_of_runs=len(runs),
                    number_of_samples=sum(run.number_of_samples for run in runs),
                    sampling_frequency=session.sampling_frequency,
                    time_vector=_runs_time_vector(runs, session.sampling_frequency, session.start_time),
                )
            )
        else:
            for run_index, run in enumerate(runs):
                self.add_recording_segment(
                    MEF3RecordingSegment(
                        channels=session.channels,
                        run_index=run_index,
                        number_of_samples=run.number_of_samples,
                        sampling_frequency=session.sampling_frequency,
                        t_start=(run.start_time - session.start_time) / 1e6,
                    )
                )

        # Gains take the stored counts to microvolts; the ecephys writer then scales microvolts to the volts
        # NWB's ElectricalSeries expects. MEF has no offset.
        self.set_channel_gains(gains=_gains_to_microvolts(session))
        self.set_channel_offsets(offsets=np.zeros(len(channel_names)))
        self.set_property(key="channel_name", values=np.asarray(channel_names))
        # The unit the metadata declares, and the scaling into it rather than into microvolts, so a caller can
        # check the metadata against an external source such as a BIDS channels.tsv, as on the EDF readers.
        self.set_property(
            key="physical_unit", values=np.asarray([channel.units_description for channel in session.channels])
        )
        self.set_property(
            key="gain_to_physical_unit",
            values=np.asarray([channel.units_conversion_factor for channel in session.channels]),
        )
        self.set_property(key="offset_to_physical_unit", values=np.zeros(len(channel_names)))
        references = [channel.reference_description for channel in session.channels]
        if any(references):
            self.set_property(key="reference", values=np.asarray(references))

        self.mef3_session = session
        self._kwargs = dict(
            folder_path=str(folder_path), channels_to_skip=channels_to_skip, concatenate_runs=concatenate_runs
        )
