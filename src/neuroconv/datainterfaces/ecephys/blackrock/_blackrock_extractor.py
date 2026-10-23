"""
SpikeInterface extractors for Blackrock file sets, over the runs and timeline of :mod:`._blackrock_file_set`.

This module is where SpikeInterface enters, so it must only be imported lazily, as the interfaces'
``get_extractor_class`` do. The classes are defined at module scope because SpikeInterface records an
extractor's class by its import path in ``to_dict()``.
"""

import warnings
from pathlib import Path

import numpy as np
from spikeinterface.core import (
    BaseRecording,
    BaseRecordingSegment,
    BaseSorting,
    BaseSortingSegment,
)

from ._blackrock_file_set import (
    BlackrockStream,
    available_nsx_files,
    file_set_base,
    nsx_number,
    read_nev_cached,
    read_stream,
    warn_about_stream,
)
from .._channel_ids import (
    _stacklevel_outside_neuroconv,
    set_original_channel_names,
    unique_channel_ids,
    warn_about_renamed_channels,
)
from .._voltage_units import unit_to_microvolts


class BlackrockRecordingSegment(BaseRecordingSegment):
    """One uninterrupted run of a ``.nsX`` stream, read from neo's memory maps, one or more data blocks long."""

    def __init__(self, *, pieces: list, number_of_samples: int, sampling_frequency: float, t_start: float):
        super().__init__(sampling_frequency=sampling_frequency, t_start=t_start)
        self._pieces = pieces
        self._number_of_samples = number_of_samples
        self._piece_ends = np.cumsum([_piece_length(data, rows) for data, rows in pieces])

    def get_num_samples(self) -> int:
        return int(self._number_of_samples)

    def get_traces(self, start_frame=None, end_frame=None, channel_indices=None) -> np.ndarray:
        start_frame = 0 if start_frame is None else max(int(start_frame), 0)
        end_frame = self._number_of_samples if end_frame is None else min(int(end_frame), self._number_of_samples)
        channel_indices = slice(None) if channel_indices is None else channel_indices
        chunks = []
        piece_start = 0
        for (data, rows), piece_end in zip(self._pieces, self._piece_ends):
            first, last = max(start_frame, piece_start), min(end_frame, piece_end)
            if first < last:
                local = slice(first - piece_start, last - piece_start)
                if rows is None:
                    selected = data[local]
                elif isinstance(rows, slice):
                    selected = data[rows][local]
                else:
                    # Packets put back in timestamp order: read the run's span of the file once, then reorder.
                    wanted = rows[local]
                    low = int(wanted.min())
                    selected = np.asarray(data[low : int(wanted.max()) + 1])[wanted - low]
                chunks.append(np.asarray(selected)[:, channel_indices])
            piece_start = piece_end
        if not chunks:
            width = len(np.empty(self._pieces[0][0].shape[1])[channel_indices])
            return np.empty((0, width), dtype=np.int16)
        return chunks[0] if len(chunks) == 1 else np.concatenate(chunks)


def _piece_length(data, rows) -> int:
    if rows is None:
        return len(data)
    if isinstance(rows, slice):
        return len(range(len(data))[rows])
    return len(rows)


def resolve_file_set(file_path, nsx_override=None, nev_file_path=None) -> tuple[Path, Path | None]:
    """
    The ``.nsX`` file to read and the ``.nev`` file recorded with it, if there is one.

    ``file_path`` is normally the ``.nsX`` file. As neo allows, it can instead name the file set (its base name,
    or its ``.nev``), with ``nsx_override`` the ``.nsX`` file when that has another base name; a base name with
    exactly one ``.nsX`` beside it means that one.
    """
    file_path = Path(file_path)
    if nsx_override is not None:
        nsx_path = Path(nsx_override)
        set_base = file_set_base(file_path)
    elif nsx_number(file_path) is not None:
        nsx_path = file_path
        set_base = file_set_base(file_path)
    else:
        set_base = file_set_base(file_path)
        candidates = available_nsx_files(set_base)
        if len(candidates) != 1:
            found = ", ".join(path.name for path in candidates) or "none"
            raise ValueError(
                f"'{file_path}' is not a Blackrock .ns1 to .ns6 file. Pass the .nsX file to convert as file_path, "
                f"or a .nev file or base name with nsx_override naming the .nsX file. .nsX files found for that "
                f"base name: {found}."
            )
        nsx_path = candidates[0]
    if nsx_number(nsx_path) is None:
        raise ValueError(f"'{nsx_path}' is not a Blackrock continuous data file: its suffix is not .ns1 to .ns6.")

    if nev_file_path is not None:
        nev_path = Path(nev_file_path)
    else:
        nev_path = Path(f"{set_base}.nev")
        if not nev_path.is_file():
            nev_path = Path(f"{file_set_base(nsx_path)}.nev")
    return nsx_path, (nev_path if nev_path.is_file() else None)


def _gains_to_microvolts(stream: BlackrockStream) -> tuple[np.ndarray, np.ndarray]:
    factors, not_voltage = [], {}
    for label, unit in zip(stream.channel_labels, stream.units):
        factor = unit_to_microvolts(unit)
        if factor is None:
            not_voltage.setdefault(unit, []).append(label)
            factor = 1.0
        factors.append(factor)
    for unit, labels in sorted(not_voltage.items()):
        declared = f"the unit {unit!r}, which is not a voltage" if unit else "no unit"
        shown = ", ".join(repr(label) for label in labels[:5]) + (f", ... ({len(labels)})" if len(labels) > 5 else "")
        warnings.warn(
            f"The channels {shown} of '{stream.path.name}' declare {declared}. Their values are written to an NWB "
            "ElectricalSeries as though they were microvolts.",
            UserWarning,
            stacklevel=_stacklevel_outside_neuroconv(),
        )
    factors = np.asarray(factors)
    return stream.gains * factors, stream.offsets * factors


class BlackrockRecordingExtractor(BaseRecording):
    """
    Lazy SpikeInterface recording for one ``.nsX`` stream of a Blackrock file set.

    Each uninterrupted run of samples is one segment, placed on the timeline of :mod:`._blackrock_file_set`:
    the earliest first sample of the file set at 0, later runs at their clock offset from it, runs after a clock
    reset where the run before ends.

    Parameters
    ----------
    file_path : str or Path
        The ``.nsX`` file, or the file set as :func:`resolve_file_set` describes.
    nsx_override : str or Path, optional
        The ``.nsX`` file, when ``file_path`` names the file set.
    nev_file_path : str or Path, optional
        The ``.nev`` file, when it does not share the ``.nsX`` file's base name.
    """

    extractor_name = "BlackrockRecording"
    mode = "file"
    name = "blackrock"

    def __init__(self, file_path, nsx_override=None, nev_file_path=None):
        nsx_path, nev_path = resolve_file_set(file_path, nsx_override=nsx_override, nev_file_path=nev_file_path)
        nev = read_nev_cached(nev_path) if nev_path is not None else None
        stream = read_stream(nsx_path, nev=nev)
        warn_about_stream(stream)

        channel_ids = unique_channel_ids(stream.channel_ids)
        super().__init__(sampling_frequency=stream.sampling_frequency, channel_ids=channel_ids, dtype=np.dtype("int16"))
        for run in stream.runs:
            self.add_recording_segment(
                BlackrockRecordingSegment(
                    pieces=run.pieces,
                    number_of_samples=run.number_of_samples,
                    sampling_frequency=stream.sampling_frequency,
                    t_start=float(run.start_time),
                )
            )

        gains, offsets = _gains_to_microvolts(stream)
        self.set_channel_gains(gains=gains)
        self.set_channel_offsets(offsets=offsets)
        # The electrodes table identifies a channel by its name, so a label the file repeats is renamed, as in the
        # other EEG readers; the ids stay the electrode ids neo and SpikeInterface give.
        names = unique_channel_ids(stream.channel_labels)
        warn_about_renamed_channels(
            labels=stream.channel_labels,
            ids=names,
            source=f"The Blackrock file '{stream.path.name}'",
            accepts_channels_to_skip=False,
        )
        self.set_property(key="channel_name", values=np.asarray(names))
        set_original_channel_names(self, labels=stream.channel_labels, ids=names)
        self.set_property(key="physical_unit", values=np.asarray(stream.units))
        self.set_property(key="gain_to_physical_unit", values=stream.gains)
        self.set_property(key="offset_to_physical_unit", values=stream.offsets)

        self.blackrock_stream = stream
        self.blackrock_nev = nev
        self._kwargs = dict(
            file_path=str(Path(file_path).absolute()),
            nsx_override=None if nsx_override is None else str(Path(nsx_override).absolute()),
            nev_file_path=None if nev_file_path is None else str(Path(nev_file_path).absolute()),
        )


class BlackrockSortingSegment(BaseSortingSegment):
    def __init__(self, *, spike_frames: np.ndarray, spike_units: np.ndarray, number_of_units: int, t_start: float):
        super().__init__(t_start=t_start)
        order = np.argsort(spike_frames, kind="stable")
        frames, units = spike_frames[order], spike_units[order]
        self._trains = [frames[units == unit_index] for unit_index in range(number_of_units)]

    def get_unit_spike_train(self, unit_id, start_frame, end_frame) -> np.ndarray:
        train = self._trains[int(unit_id)]
        if start_frame is not None:
            train = train[train >= start_frame]
        if end_frame is not None:
            train = train[train < end_frame]
        return train


def select_timeline_stream(nev, nsx_to_load=None) -> BlackrockStream | None:
    """
    The ``.nsX`` stream whose timeline places a ``.nev`` file's spikes, among those ``nsx_to_load`` allows.

    Every stream of a set is placed from the set's earliest first sample, so any of them gives the spikes the
    same times, up to which stream's samples a PTP spike is anchored to and where a run after a clock reset ends.
    Of several, the smallest file is read, as the cheapest; ``None`` when there is no ``.nsX`` to use.
    """
    candidates = available_nsx_files(file_set_base(nev.path))
    if nsx_to_load is not None and nsx_to_load not in ("all", "max"):
        wanted = {int(nsx_to_load)} if isinstance(nsx_to_load, int) else {int(number) for number in nsx_to_load}
        candidates = [path for path in candidates if nsx_number(path) in wanted]
    elif nsx_to_load == "max" and candidates:
        candidates = candidates[-1:]
    if not candidates:
        return None
    return read_stream(min(candidates, key=lambda path: path.stat().st_size), nev=nev)


class BlackrockSortingExtractor(BaseSorting):
    """
    The spikes of a Blackrock ``.nev`` file, on the timeline of the ``.nsX`` stream recorded with it.

    Unit ids are 0, 1, ... in neo's order, as SpikeInterface's Blackrock sorting gives them, with
    ``electrode_id`` and ``unit_number`` (the ``.nev`` unit classification number, 0 for unsorted) properties.

    Parameters
    ----------
    file_path : str or Path
        The ``.nev`` file.
    sampling_frequency : float, optional
        The rate spike frames are counted at; by default the ``.nev`` header's sample resolution.
    nsx_to_load : int, list, 'max' or 'all', optional
        The ``.nsX`` streams whose timeline may place the spikes; see :func:`select_timeline_stream`. An empty
        list uses none, which places the spikes on the ``.nev`` clock, its first packet at 0 for FileSpec 3.0.
    """

    extractor_name = "BlackrockSorting"
    mode = "file"
    name = "blackrock_sorting"

    def __init__(self, file_path, sampling_frequency=None, nsx_to_load=None):
        nev_path = Path(file_path)
        nev = read_nev_cached(nev_path)
        stream = select_timeline_stream(nev, nsx_to_load=nsx_to_load)
        sampling_frequency = float(sampling_frequency) if sampling_frequency else nev.sample_resolution

        if stream is not None:
            warn_about_stream(stream)
            seconds = stream.timeline.to_seconds(nev.spike_ticks, nev.resolution, nev.spike_epochs)
        else:
            seconds = _nev_clock_seconds(nev)
        unit_index = {key: index for index, key in enumerate(nev.unit_keys)}
        units = np.asarray(
            [unit_index.get((electrode, unit), -1) for electrode, unit in zip(nev.spike_electrodes, nev.spike_units)],
            dtype=int,
        )
        kept = (units >= 0) & np.isfinite(seconds)
        if np.count_nonzero(~np.isfinite(seconds)):
            warnings.warn(
                f"{np.count_nonzero(~np.isfinite(seconds))} spikes of '{nev_path.name}' fall after a clock reset "
                f"that its .nsX stream does not have, so their time is unknown; they are left out.",
                UserWarning,
                stacklevel=_stacklevel_outside_neuroconv(),
            )
        seconds, units = seconds[kept], units[kept]
        t_start = min(0.0, float(seconds.min())) if len(seconds) else 0.0

        super().__init__(sampling_frequency=sampling_frequency, unit_ids=list(range(len(nev.unit_keys))))
        self.add_sorting_segment(
            BlackrockSortingSegment(
                spike_frames=np.round((seconds - t_start) * sampling_frequency).astype(np.int64),
                spike_units=units,
                number_of_units=len(nev.unit_keys),
                t_start=t_start,
            )
        )
        self.set_property(key="electrode_id", values=np.asarray([key[0] for key in nev.unit_keys], dtype=int))
        self.set_property(key="unit_number", values=np.asarray([key[1] for key in nev.unit_keys], dtype=int))

        self.blackrock_nev = nev
        self.blackrock_stream = stream
        self._kwargs = dict(
            file_path=str(nev_path.absolute()), sampling_frequency=sampling_frequency, nsx_to_load=nsx_to_load
        )


def _nev_clock_seconds(nev) -> np.ndarray:
    """
    Spike times with no ``.nsX`` to place them by: the ``.nev``'s own clock, shifted past each reset.

    A FileSpec 2.x clock starts with the recording, so it is used as it is; a FileSpec 3.0 one does not, so its
    first packet is put at 0. A reset is handled as for a stream, the packets after it placed after the last
    packet before it.
    """
    ticks = nev.spike_ticks.astype(np.int64)
    if not len(ticks):
        return np.zeros(0)
    reference = int(ticks.min()) if nev.spec == "3.0" else 0
    seconds = np.empty(len(ticks))
    base = 0.0
    for epoch in np.unique(nev.spike_epochs):
        mask = nev.spike_epochs == epoch
        epoch_reference = reference if epoch == 0 else 0
        seconds[mask] = base + (ticks[mask] - epoch_reference) / nev.resolution
        base = float(seconds[mask].max())
    return seconds
