"""
Support for EDF+ signals that carry floating-point or long-integer data.

EDF stores every sample as a 2-byte integer, which cannot hold the dynamic range of floating-point or
long-integer data. EDF+ addresses that with a logarithmic transformation rather than a wider sample: the
samples stay 2-byte integers, the signal is marked by the physical dimension ``Filtered``, and its
original unit together with the two parameters go in the prefiltering field, as
``sign*LN[sign*(uV      )/(0.01    )]/(0.002   )``. See https://www.edfplus.info/specs/edffloat.html.

SpikeInterface's reader has no notion of the transformation. It opens such a file *successfully* and
applies the header's linear gain and offset to what are logarithms, so values come back wrong by many
orders of magnitude with no error and no warning — a signal spanning 0.05 to 5e6 µV reads back as 24 to
306 µV. This module recovers the physical values by decoding lazily on top of that reader, so nothing
about how the file is opened changes.
"""

import re
import warnings
from typing import NamedTuple

import numpy as np
from pydantic import FilePath

# The physical dimension EDF+ reserves to mark a logarithmically transformed signal.
_LOG_TRANSFORM_DIMENSION = "FILTERED"

# The prefiltering field of such a signal holds the original unit, Ymin and a.
_LOG_TRANSFORM_PATTERN = re.compile(
    r"sign\s*\*\s*LN\s*\[\s*sign\s*\*\s*\(([^)]*)\)\s*/\s*\(([^)]*)\)\s*\]\s*/\s*\(([^)]*)\)",
    re.IGNORECASE,
)

# Byte geometry of the EDF header. The static part is followed by one block per field, each holding one
# fixed-width entry per signal, in this order.
_STATIC_HEADER_SIZE = 256
_NUMBER_OF_SIGNALS_OFFSET = 252
_NUMBER_OF_SIGNALS_WIDTH = 4
_DIMENSION_WIDTH = 8
_PREFILTER_WIDTH = 80
# Widths of the per-signal fields preceding each of the two this module needs.
_WIDTHS_BEFORE_DIMENSION = (16, 80)  # label, transducer
_WIDTHS_BEFORE_PREFILTER = (16, 80, 8, 8, 8, 8, 8)  # ... dimension, physical/digital min and max

# Voltage units, written out rather than taken from utils.units.get_conversion_from_unit, whose contract
# is volts and whose fallback for an unrecognized unit is 1.0 — "assume volts" — which here would scale a
# decoded signal by 1e6. These are exact powers of ten.
_VOLTAGE_UNIT_TO_MICROVOLTS = {
    "V": 1e6,
    "VOLT": 1e6,
    "VOLTS": 1e6,
    "MV": 1e3,
    "MILLIVOLT": 1e3,
    "MILLIVOLTS": 1e3,
    "UV": 1.0,
    "MICROVOLT": 1.0,
    "MICROVOLTS": 1.0,
    "NV": 1e-3,
    "NANOVOLT": 1e-3,
    "NANOVOLTS": 1e-3,
}


class LogTransform(NamedTuple):
    """
    Parameters of the EDF+ logarithmic transform for one signal.

    ``unit`` is the signal's *original* physical dimension, which the transform displaces from the
    header's own dimension field. Decoding gives values back in that unit.
    """

    unit: str
    minimum: float  # Ymin: the smallest magnitude representable; below it, samples are 0
    coefficient: float  # a: the logarithmic step, so accuracy is roughly a/2 in relative terms

    def decode(self, counts) -> np.ndarray:
        """
        Recover physical values from stored integer counts.

        ``Y = Ymin*exp(a*N)`` for ``N > 0``, ``0`` for ``N == 0``, and ``-Ymin*exp(-a*N)`` for ``N < 0``.
        """
        counts = np.asarray(counts)
        values = np.zeros(counts.shape, dtype="float64")
        positive = counts > 0
        negative = counts < 0
        values[positive] = self.minimum * np.exp(self.coefficient * counts[positive].astype("float64"))
        values[negative] = -self.minimum * np.exp(-self.coefficient * counts[negative].astype("float64"))
        return values


def _decode_field(raw: bytes) -> str:
    """
    Decode one EDF header field.

    latin-1 rather than ASCII: every byte maps to a character, so the micro sign a writer may put in a
    dimension field (byte 0xB5, outside ASCII) survives instead of becoming a replacement character.
    ASCII fields decode identically. NUL is treated as padding, which some writers use instead of spaces.
    """
    return raw.decode("latin-1").replace("\x00", " ").strip()


def read_signal_dimensions_and_prefilters(file_path: FilePath) -> tuple[list[str], list[str]]:
    """
    Read just the physical dimension and prefiltering fields of every signal.

    SpikeInterface exposes the dimension as the ``physical_unit`` channel property but not the
    prefiltering field, which is where the transform's parameters live, so both are read here — reading
    both keeps this independent of that property's name.

    Parameters
    ----------
    file_path : FilePath
        Path to the ``.edf`` file.

    Returns
    -------
    tuple of list of str
        ``(dimensions, prefilters)``, one entry per signal in file order.
    """
    with open(file_path, "rb") as file:
        file.seek(_NUMBER_OF_SIGNALS_OFFSET)
        number_of_signals = int(_decode_field(file.read(_NUMBER_OF_SIGNALS_WIDTH)) or 0)
        if number_of_signals <= 0:
            return [], []

        def read_block(widths_before: tuple, width: int) -> list[str]:
            file.seek(_STATIC_HEADER_SIZE + number_of_signals * sum(widths_before))
            return [_decode_field(file.read(width)) for _ in range(number_of_signals)]

        dimensions = read_block(_WIDTHS_BEFORE_DIMENSION, _DIMENSION_WIDTH)
        prefilters = read_block(_WIDTHS_BEFORE_PREFILTER, _PREFILTER_WIDTH)
    return dimensions, prefilters


def _parse_log_transform(dimension: str, prefilter: str) -> LogTransform | None:
    """
    Return the logarithmic transform for a signal, or None if it is an ordinary linear one.

    Raises
    ------
    ValueError
        If the signal is marked as transformed but the parameters cannot be read. Its samples are
        logarithms rather than measurements, so without ``Ymin`` and ``a`` they cannot be interpreted at
        all, and guessing would write values wrong by orders of magnitude.
    """
    if dimension.strip().upper() != _LOG_TRANSFORM_DIMENSION:
        return None

    match = _LOG_TRANSFORM_PATTERN.search(prefilter or "")
    if match:
        unit, minimum_text, coefficient_text = (group.strip() for group in match.groups())
        try:
            minimum, coefficient = float(minimum_text), float(coefficient_text)
        except ValueError:
            minimum = coefficient = 0.0
        if minimum > 0 and coefficient > 0:
            return LogTransform(unit=unit, minimum=minimum, coefficient=coefficient)

    raise ValueError(
        "This EDF signal has the physical dimension 'Filtered', marking it as logarithmically transformed "
        "(https://www.edfplus.info/specs/edffloat.html), but its prefiltering field does not carry readable "
        f"parameters: {prefilter!r}. Expected the form 'sign*LN[sign*(uV      )/(0.01    )]/(0.002   )'. "
        "Without Ymin and a the stored samples cannot be interpreted — they are logarithms rather than "
        "measurements — so the signal is refused instead of being written to NWB wrong by orders of "
        "magnitude. Use channels_to_skip to drop it if you do not need it."
    )


def read_log_transforms(file_path: FilePath, channels_to_skip: list | None = None) -> dict[str, LogTransform]:
    """
    Map channel label to logarithmic transform for every transformed signal in a file.

    Parameters
    ----------
    file_path : FilePath
        Path to the ``.edf`` file.
    channels_to_skip : list, optional
        Labels to leave out entirely. Applied before the parameters are parsed, so a transformed signal
        whose parameters are unreadable can be dropped rather than failing the whole conversion — the
        caller's own channel removal happens too late for that.

    Returns
    -------
    dict
        Empty when no signal is transformed, which is the overwhelmingly common case.
    """
    dimensions, prefilters = read_signal_dimensions_and_prefilters(file_path=file_path)
    if not any(dimension.strip().upper() == _LOG_TRANSFORM_DIMENSION for dimension in dimensions):
        return {}

    with open(file_path, "rb") as file:
        file.seek(_NUMBER_OF_SIGNALS_OFFSET)
        number_of_signals = int(_decode_field(file.read(_NUMBER_OF_SIGNALS_WIDTH)) or 0)
        file.seek(_STATIC_HEADER_SIZE)
        labels = [_decode_field(file.read(16)) for _ in range(number_of_signals)]

    transforms = {}
    skip = set(channels_to_skip or [])
    for label, dimension, prefilter in zip(labels, dimensions, prefilters):
        if label in skip:
            continue
        transform = _parse_log_transform(dimension=dimension, prefilter=prefilter)
        if transform is not None:
            transforms[label] = transform
    return transforms


def _unit_to_microvolts(unit: str, channel_name: str) -> float:
    """
    Factor converting a decoded signal's original unit to microvolts.

    A unit that is not a voltage has no correct factor for an ``ElectricalSeries``, so microvolts is
    assumed — the least-wrong assumption, and unlike "assume volts" it cannot inflate a value by 1e6 —
    and a warning says so.
    """
    normalized = str(unit or "").strip().replace("µ", "u").replace("μ", "u").upper()
    factor = _VOLTAGE_UNIT_TO_MICROVOLTS.get(normalized)
    if factor is not None:
        return factor

    warnings.warn(
        f"The transformed EDF channel {channel_name!r} declares the unit {unit!r}, which is not a voltage. "
        "Its decoded values are being written to an NWB ElectricalSeries as though they were microvolts, "
        "so they will be wrong by whatever factor separates that unit from microvolts. Drop the channel "
        "with channels_to_skip and convert it with EDFAnalogInterface if it is not neural data.",
        UserWarning,
        stacklevel=3,
    )
    return 1.0


def get_log_transform_recording_class():
    """Return the decoding recording class, importing SpikeInterface on first call."""
    from spikeinterface.preprocessing.basepreprocessor import (
        BasePreprocessor,
        BasePreprocessorSegment,
    )

    class EDFLogTransformRecordingSegment(BasePreprocessorSegment):
        """Decodes the transformed channels of one segment as traces are requested."""

        def __init__(self, parent_recording_segment, transform_by_position: dict, number_of_channels: int):
            super().__init__(parent_recording_segment)
            self._transform_by_position = transform_by_position
            # Passed in rather than discovered at read time: a parent segment does not hold a reference
            # back to its recording, and the count cannot change once the wrapper is built.
            self._number_of_channels = number_of_channels

        def get_traces(self, start_frame, end_frame, channel_indices) -> np.ndarray:
            if channel_indices is None:
                channel_indices = slice(None)
            parent_traces = self.parent_recording_segment.get_traces(
                start_frame=start_frame, end_frame=end_frame, channel_indices=channel_indices
            )
            positions = np.arange(self._number_of_channels)[channel_indices]

            traces = np.asarray(parent_traces, dtype="float64")
            for column, position in enumerate(np.atleast_1d(positions)):
                transform = self._transform_by_position.get(int(position))
                if transform is not None:
                    traces[:, column] = transform.decode(parent_traces[:, column])
            return traces

    class EDFLogTransformRecording(BasePreprocessor):
        """
        Decodes an EDF recording's logarithmically transformed channels.

        Wrapping SpikeInterface's reader rather than replacing it: the file is opened exactly as before,
        and only the marked channels' samples are reinterpreted. Decoding happens here rather than through
        the recording's gain and offset because the relationship between stored integer and physical value
        is exponential, while an ``ElectricalSeries`` applies only a scalar gain and offset.

        The dtype becomes ``float64`` for the whole recording, since a recording carries one dtype and the
        specification's own accuracy table reaches 1.4e68 — past what ``float32`` holds.
        """

        name = "edf_log_transform"

        def __init__(self, recording, transforms: dict):
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

            # A decoded channel's values are already physical, so its gain is only the unit scaling and
            # its offset is zero. Untransformed channels keep whatever the parent reader worked out.
            gains = np.asarray(recording.get_channel_gains(), dtype="float64").copy()
            offsets = np.asarray(recording.get_channel_offsets(), dtype="float64").copy()
            for position, transform in transform_by_position.items():
                gains[position] = _unit_to_microvolts(unit=transform.unit, channel_name=str(channel_ids[position]))
                offsets[position] = 0.0
            self.set_channel_gains(gains=gains)
            self.set_channel_offsets(offsets=offsets)

            self._kwargs = dict(recording=recording, transforms=transforms)

    return EDFLogTransformRecording


def decode_log_transformed_signals(recording, file_path: FilePath, channels_to_skip: list | None = None):
    """
    Wrap a recording so its logarithmically transformed channels decode to physical values.

    Returns the recording unchanged when the file has no transformed signal, so the ordinary path is
    untouched.

    Parameters
    ----------
    recording : BaseRecording
        The recording as SpikeInterface read it.
    file_path : FilePath
        Path to the ``.edf`` file it came from.
    channels_to_skip : list, optional
        Labels to leave out of the transform lookup entirely.

    Returns
    -------
    tuple
        ``(recording, transforms)``, the latter mapping channel label to its transform.
    """
    transforms = read_log_transforms(file_path=file_path, channels_to_skip=channels_to_skip)
    present = {label: transform for label, transform in transforms.items() if label in set(recording.get_channel_ids())}
    if not present:
        return recording, transforms
    return get_log_transform_recording_class()(recording=recording, transforms=present), transforms
