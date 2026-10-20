"""
Factors converting the voltage units that EEG file headers declare into microvolts.

Shared by the readers that set channel gains from a header's unit string (EDF, BDF, BrainVision), so a
spelling one of them learns to recognise reaches all of them.

Written out rather than taken from ``utils.units.get_conversion_from_unit``, whose contract is volts and
whose fallback for an unrecognized unit is 1.0 ("assume volts"), which here would scale a signal by 1e6.
These are exact powers of ten.
"""

VOLTAGE_UNIT_TO_MICROVOLTS = {
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


def unit_to_microvolts(unit: str) -> float | None:
    """
    Factor converting a value in ``unit`` to microvolts, or None if ``unit`` is not a voltage.

    Matched case-insensitively, with both the micro sign (U+00B5) and the Greek mu (U+03BC) read as ``u``.
    Returning None rather than assuming lets the caller decide what an unrecognized unit means and
    aggregate its warnings, so a file whose channels share an odd unit warns once rather than per channel.
    """
    normalized = str(unit or "").strip().replace("µ", "u").replace("μ", "u").upper()
    return VOLTAGE_UNIT_TO_MICROVOLTS.get(normalized)
