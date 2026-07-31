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
_LABEL_WIDTH = 16
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
    "MVOLT": 1e3,
    "MVOLTS": 1e3,
    "MILLIVOLT": 1e3,
    "MILLIVOLTS": 1e3,
    "UV": 1.0,
    "UVOLT": 1.0,
    "UVOLTS": 1.0,
    "MICROV": 1.0,
    "MICROVOLT": 1.0,
    "MICROVOLTS": 1.0,
    "NV": 1e-3,
    "NVOLT": 1e-3,
    "NVOLTS": 1e-3,
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


def read_signal_fields(file_path: FilePath) -> tuple[list[str], list[str], list[str]]:
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
        ``(labels, dimensions, prefilters)``, one entry per signal in file order.
    """
    with open(file_path, "rb") as file:
        file.seek(_NUMBER_OF_SIGNALS_OFFSET)
        number_of_signals = int(_decode_field(file.read(_NUMBER_OF_SIGNALS_WIDTH)) or 0)
        if number_of_signals <= 0:
            return [], [], []

        def read_block(widths_before: tuple, width: int) -> list[str]:
            file.seek(_STATIC_HEADER_SIZE + number_of_signals * sum(widths_before))
            return [_decode_field(file.read(width)) for _ in range(number_of_signals)]

        labels = read_block((), _LABEL_WIDTH)
        dimensions = read_block(_WIDTHS_BEFORE_DIMENSION, _DIMENSION_WIDTH)
        prefilters = read_block(_WIDTHS_BEFORE_PREFILTER, _PREFILTER_WIDTH)
    return labels, dimensions, prefilters


def filtered_accounts_for_the_whole_unit_mix(dimensions: list) -> bool:
    """
    Whether ``Filtered`` is the only reason a file looks like it mixes voltage and non-voltage units.

    SpikeInterface warns about such a mix, and for a transformed signal that warning is unactionable and,
    once decoded, untrue. But a file can carry both a transformed signal *and* a genuinely non-voltage
    channel, and that warning is the only thing telling the user about the latter — so it may only be
    suppressed when nothing else accounts for the mix.

    Parameters
    ----------
    dimensions : list of str
        Every signal's physical dimension, including the annotations signal's empty one.

    Returns
    -------
    bool
    """
    for dimension in dimensions:
        normalized = str(dimension or "").strip()
        if not normalized:  # the annotations signal carries no unit
            continue
        if normalized.upper() == _LOG_TRANSFORM_DIMENSION:
            continue
        if _unit_to_microvolts(unit=normalized) is None:
            return False
    return True


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
    labels, dimensions, prefilters = read_signal_fields(file_path=file_path)
    if not any(dimension.strip().upper() == _LOG_TRANSFORM_DIMENSION for dimension in dimensions):
        return {}

    transforms = {}
    skip = set(channels_to_skip or [])
    for label, dimension, prefilter in zip(labels, dimensions, prefilters):
        if label in skip:
            continue
        transform = _parse_log_transform(dimension=dimension, prefilter=prefilter)
        if transform is not None:
            transforms[label] = transform
    return transforms


def _unit_to_microvolts(unit: str) -> float | None:
    """
    Factor converting a decoded signal's original unit to microvolts, or None if it is not a voltage.

    Returning None rather than assuming here lets the caller aggregate, so a file whose transformed
    channels share an odd unit warns once rather than once per channel.
    """
    normalized = str(unit or "").strip().replace("µ", "u").replace("μ", "u").upper()
    return _VOLTAGE_UNIT_TO_MICROVOLTS.get(normalized)


def _warn_assumed_microvolts(unit_to_channel_names: dict) -> None:
    """
    Report each non-voltage unit whose channels were read as microvolts anyway.

    Microvolts is the least-wrong assumption for an ``ElectricalSeries`` — and unlike "assume volts" it
    cannot inflate a value by 1e6 — but it is still an assumption, so it is stated. One warning per
    distinct unit, since the message is identical for every channel sharing one.
    """
    for unit, channel_names in sorted(unit_to_channel_names.items()):
        shown = ", ".join(repr(name) for name in channel_names[:5])
        if len(channel_names) > 5:
            shown += f", ... ({len(channel_names)} channels)"
        warnings.warn(
            f"The transformed EDF channels {shown} declare the unit {unit!r}, which is not a voltage. Their "
            "decoded values are being written to an NWB ElectricalSeries as though they were microvolts, so "
            "they will be wrong by whatever factor separates that unit from microvolts. Drop them with "
            "channels_to_skip and convert them with EDFAnalogInterface if they are not neural data.",
            UserWarning,
            stacklevel=4,
        )


def get_log_transform_recording_class():
    """
    Return the decoding recording class, importing SpikeInterface on first call.

    The class lives at module scope in ``_edf_log_transform_recording`` rather than being built here:
    SpikeInterface records an extractor's class by import path, so a class defined inside this function
    could not be pickled or reloaded, and two calls would return distinct class objects.
    """
    from ._edf_log_transform_recording import EDFLogTransformRecording

    return EDFLogTransformRecording


def decode_log_transformed_signals(recording, transforms: dict):
    """
    Wrap a recording so its logarithmically transformed channels decode to physical values.

    Returns the recording unchanged when none of the transforms names a channel it actually has, so the
    ordinary path is untouched.

    Parameters
    ----------
    recording : BaseRecording
        The recording as SpikeInterface read it.
    transforms : dict
        Channel label to transform, as returned by :func:`read_log_transforms`. Note that it may name a
        signal the recording does not expose, so it is not necessarily a subset of its channel ids.

    Returns
    -------
    tuple
        ``(recording, transforms)``.
    """
    present = {label: transform for label, transform in transforms.items() if label in set(recording.get_channel_ids())}
    if not present:
        return recording, transforms
    return get_log_transform_recording_class()(recording=recording, transforms=present), transforms
