"""
The SpikeInterface recording that decodes logarithmically transformed EDF+ signals.

Kept apart from :mod:`._edf_log_transform`, which holds the parsing and depends only on NumPy, because
this module imports SpikeInterface — an optional ("ecephys") extra — so it must only ever be imported
lazily, from inside a function.

The classes are defined at module scope on purpose. SpikeInterface records an extractor's class by its
import path in ``to_dict()``, so classes built inside a factory function cannot be pickled or reloaded,
which breaks ``recording.save()`` and every ``n_jobs > 1`` path.
"""

import numpy as np
from spikeinterface.preprocessing.basepreprocessor import (
    BasePreprocessor,
    BasePreprocessorSegment,
)

from ._edf_log_transform import (
    LogTransform,
    _unit_to_microvolts,
    _warn_assumed_microvolts,
)


def _as_log_transform(value) -> LogTransform:
    """
    Coerce a stored transform back into a :class:`LogTransform`.

    A provenance round-trip through JSON flattens a NamedTuple to a plain list, so a reloaded recording
    hands back ``["uV", 0.01, 0.002]`` where the original held the tuple. Accepting every shape keeps
    ``to_dict()`` / ``load()`` working without making the caller aware of the serialization.
    """
    if isinstance(value, LogTransform):
        return value
    if isinstance(value, dict):
        return LogTransform(**value)
    return LogTransform(*value)


class EDFLogTransformRecordingSegment(BasePreprocessorSegment):
    """Decodes the transformed channels of one segment as traces are requested."""

    def __init__(self, parent_recording_segment, transform_by_position: dict, number_of_channels: int):
        super().__init__(parent_recording_segment)
        self._transform_by_position = transform_by_position
        # Passed in rather than discovered at read time: a parent segment holds no reference back to its
        # recording, and the count cannot change once the wrapper is built.
        self._number_of_channels = number_of_channels

    def get_traces(self, start_frame, end_frame, channel_indices) -> np.ndarray:
        if channel_indices is None:
            channel_indices = slice(None)
        parent_traces = self.parent_recording_segment.get_traces(
            start_frame=start_frame, end_frame=end_frame, channel_indices=channel_indices
        )
        positions = np.atleast_1d(np.arange(self._number_of_channels)[channel_indices])

        traces = np.asarray(parent_traces, dtype="float64")
        for column, position in enumerate(positions):
            transform = self._transform_by_position.get(int(position))
            if transform is not None:
                traces[:, column] = transform.decode(parent_traces[:, column])
        return traces


class EDFLogTransformRecording(BasePreprocessor):
    """
    Decodes an EDF recording's logarithmically transformed channels.

    Wraps SpikeInterface's reader rather than replacing it: the file is opened exactly as before, and
    only the marked channels' samples are reinterpreted. Decoding happens here rather than through the
    recording's gain and offset because the relationship between stored integer and physical value is
    exponential, while an ``ElectricalSeries`` applies only a scalar gain and offset.

    The dtype becomes ``float64`` for the whole recording, since a recording carries one dtype and the
    specification's own accuracy table reaches 1.4e68 — past what ``float32`` holds.

    Parameters
    ----------
    recording : BaseRecording
        The recording as SpikeInterface read it.
    transforms : dict
        Channel label to its transform, as a :class:`LogTransform`, a dict of its fields, or the list a
        JSON provenance round-trip produces.
    """

    name = "edf_log_transform"

    def __init__(self, recording, transforms: dict):
        transforms = {label: _as_log_transform(value) for label, value in transforms.items()}

        # float64, not the parent's int16: decoded values are physical, not counts.
        super().__init__(recording, dtype="float64")

        channel_ids = list(recording.get_channel_ids())
        transform_by_position = {
            channel_ids.index(label): transform for label, transform in transforms.items() if label in channel_ids
        }
        for segment in recording._recording_segments:
            self.add_recording_segment(
                EDFLogTransformRecordingSegment(
                    parent_recording_segment=segment,
                    transform_by_position=transform_by_position,
                    number_of_channels=len(channel_ids),
                )
            )

        # A decoded channel's values are already physical, so its gain is only the unit scaling and its
        # offset is zero. Untransformed channels keep whatever the parent reader worked out.
        gains = np.asarray(recording.get_channel_gains(), dtype="float64").copy()
        offsets = np.asarray(recording.get_channel_offsets(), dtype="float64").copy()
        assumed_microvolts: dict = {}
        for position, transform in transform_by_position.items():
            factor = _unit_to_microvolts(unit=transform.unit)
            if factor is None:
                assumed_microvolts.setdefault(transform.unit, []).append(str(channel_ids[position]))
                factor = 1.0
            gains[position] = factor
            offsets[position] = 0.0
        self.set_channel_gains(gains=gains)
        self.set_channel_offsets(offsets=offsets)
        _warn_assumed_microvolts(unit_to_channel_names=assumed_microvolts)

        # Stored as plain dicts rather than NamedTuples: a JSON round-trip would flatten the tuples to
        # lists, and a dict survives it recognizably.
        self._kwargs = dict(
            recording=recording, transforms={label: dict(t._asdict()) for label, t in transforms.items()}
        )
