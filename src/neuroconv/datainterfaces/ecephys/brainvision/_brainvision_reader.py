"""
Reader for the BrainVision Core Data Format (``.vhdr`` header, ``.vmrk`` markers, ``.eeg`` binary data).

Pure Python and NumPy; SpikeInterface enters only in :mod:`._brainvision_extractor`.

neo's ``BrainVisionRawIO`` exists but is not used, because on real files it reads data wrong without
raising: it ignores the channel resolution whenever the samples are floating point (so a pybv-written
file with resolution 0.1 comes out ten times too large), opens every text file as UTF-8 (so AnyWave's
Latin-1 exports raise ``UnicodeDecodeError``), takes marker positions as 0-based although the format
counts from 1, drops marker sizes and the date carried by ``New Segment`` markers, and always reports a
single segment. Its unit handling passes ``nV`` through unscaled.

The format is two INI-style text files and a flat binary array, so it is read directly here. Format
rules not spelled out in the files themselves follow the two maintained BSD-licensed implementations,
pybv (the writer BIDS tooling uses) and MNE-Python's reader.
"""

import re
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

# The binary formats of the core specification, little-endian.
_BINARY_FORMAT_TO_DTYPE = {"INT_16": "<i2", "INT_32": "<i4", "IEEE_FLOAT_32": "<f4"}

# Units are given per channel. Anything that is not a voltage is still read (as microvolts, with a
# warning), since dropping it is the caller's decision.
_VOLTAGE_UNIT_TO_MICROVOLTS = {"V": 1e6, "MV": 1e3, "UV": 1.0, "NV": 1e-3}

# Writers that anonymise a recording replace the New Segment date with the earliest date the field can
# hold rather than removing it (Wonambi writes 19000101000000000000). It is not a recording time.
_PLACEHOLDER_DATES = ("19000101000000000000", "00000000000000000000")

_NEW_SEGMENT = "New Segment"


@dataclass(frozen=True)
class BrainVisionMarker:
    """One entry of a ``.vmrk`` file, with the position converted to a 0-based sample index."""

    marker_type: str
    description: str
    sample_index: int
    size: int
    channel_number: int
    date: datetime | None = None


@dataclass(frozen=True)
class BrainVisionSegment:
    """A run of samples recorded without interruption, delimited in the file by ``New Segment`` markers."""

    first_sample: int
    number_of_samples: int
    start_time: float
    date: datetime | None = None


@dataclass
class BrainVisionHeader:
    header_path: Path
    data_path: Path
    marker_path: Path | None
    sampling_frequency: float
    dtype: str
    orientation: str
    number_of_samples: int
    channel_names: list[str]
    references: list[str]
    resolutions: np.ndarray
    units: list[str]
    gains_to_microvolts: np.ndarray
    markers: list[BrainVisionMarker] = field(default_factory=list)
    segments: list[BrainVisionSegment] = field(default_factory=list)

    @property
    def number_of_channels(self) -> int:
        return len(self.channel_names)

    @property
    def recording_start(self) -> datetime | None:
        """Wall-clock time of the first sample, when the marker file states one."""
        return self.segments[0].date if self.segments else None


def _decode_text(raw: bytes, file_path: Path) -> str:
    """
    Decode a header or marker file using its declared ``Codepage``.

    Files written before the format adopted UTF-8 declare ``ANSI`` (Windows-1252) or nothing at all;
    AnyWave's exporter, for one, writes the micro sign as the single Latin-1 byte 0xB5 with no
    ``Codepage`` line. Absent a declaration UTF-8 is tried first, since it fails loudly on such a byte,
    and Windows-1252 second.
    """
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]

    # The key itself is ASCII in every encoding, so it can be found before the encoding is known.
    declared = re.search(rb"^\s*Codepage\s*=\s*([^\r\n;]+)", raw, re.IGNORECASE | re.MULTILINE)
    candidates = []
    if declared:
        codepage = declared.group(1).decode("ascii", "replace").strip()
        candidates.append("cp1252" if codepage.upper() == "ANSI" else codepage)
    candidates += ["utf-8", "cp1252", "latin-1"]

    for encoding in candidates:
        try:
            return raw.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    raise AssertionError("latin-1 decodes any byte sequence")  # pragma: no cover


def _parse_ini(text: str) -> dict[str, dict[str, str]]:
    """
    Parse BrainVision's INI dialect into ``{section: {key: value}}``, both matched case-insensitively.

    Not ``configparser``: values legitimately contain ``=`` and ``;`` (marker descriptions do), and only a
    ``;`` at the start of a line begins a comment. Splitting stops at ``[Comment]``, which holds free text.
    """
    sections: dict[str, dict[str, str]] = {}
    current = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(";"):
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            name = stripped[1:-1].strip().lower()
            if name == "comment":
                break
            current = sections.setdefault(name, {})
            continue
        if current is None or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        current[key.strip().lower()] = value.strip()
    return sections


def _read_ini(file_path: Path) -> dict[str, dict[str, str]]:
    return _parse_ini(_decode_text(file_path.read_bytes(), file_path=file_path))


def _resolve_companion(header_path: Path, named: str | None, suffix: str, role: str) -> Path | None:
    """
    Find the data or marker file a header names, falling back to the header's own base name.

    Files are routinely renamed after writing (BIDS conversion does it to every file), and only the
    header's references go stale; the file itself is still beside the header under the matching name.
    """
    if named:
        path = header_path.parent / named
        if path.is_file():
            return path
    sibling = header_path.with_suffix(suffix)
    if sibling.is_file():
        if named:
            warnings.warn(
                f"The BrainVision header '{header_path.name}' names the {role} file '{named}', which does not "
                f"exist; reading '{sibling.name}' beside it instead.",
                UserWarning,
                stacklevel=4,
            )
        return sibling
    return None


def _parse_unit(unit: str) -> str:
    """Normalise a declared unit, defaulting to microvolts as the format does when none is given."""
    unit = unit.strip()
    # A UTF-8 micro sign decoded as Windows-1252 reads "Âµ"; a file that declares ANSI but was written as
    # UTF-8 produces exactly that.
    unit = unit.replace("Âµ", "µ")
    return unit or "µV"


def _unit_to_microvolts(unit: str) -> float | None:
    normalized = unit.replace("µ", "u").replace("μ", "u").upper()
    return _VOLTAGE_UNIT_TO_MICROVOLTS.get(normalized)


def _parse_channels(channel_infos: dict[str, str], number_of_channels: int, header_path: Path):
    names, references, resolutions, units = [], [], [], []
    blank_resolutions = []
    for index in range(1, number_of_channels + 1):
        entry = channel_infos.get(f"ch{index}")
        if entry is None:
            raise ValueError(
                f"The BrainVision header '{header_path.name}' declares {number_of_channels} channels but has "
                f"no entry for Ch{index} in [Channel Infos]."
            )
        fields = entry.split(",")
        fields += [""] * (4 - len(fields))
        name, reference, resolution, unit = (value.strip() for value in fields[:4])
        # Commas inside a name or reference are written as the two characters "\1".
        names.append(name.replace(r"\1", ","))
        references.append(reference.replace(r"\1", ","))
        if resolution == "":
            blank_resolutions.append(name)
            resolution = "1"
        resolutions.append(float(resolution))
        units.append(_parse_unit(unit))

    if blank_resolutions:
        warnings.warn(
            f"The BrainVision header '{header_path.name}' gives no resolution for {len(blank_resolutions)} "
            f"channel(s) ({', '.join(repr(name) for name in blank_resolutions[:5])}); their samples are taken "
            "to be in the declared unit already (resolution 1).",
            UserWarning,
            stacklevel=4,
        )
    return names, references, np.asarray(resolutions, dtype="float64"), units


def _gains_to_microvolts(resolutions: np.ndarray, units: list[str], names: list[str]) -> np.ndarray:
    gains = np.empty(len(units), dtype="float64")
    non_voltage: dict[str, list[str]] = {}
    for index, unit in enumerate(units):
        factor = _unit_to_microvolts(unit)
        if factor is None:
            non_voltage.setdefault(unit, []).append(names[index])
            factor = 1.0
        gains[index] = resolutions[index] * factor

    for unit, channel_names in sorted(non_voltage.items()):
        shown = ", ".join(repr(name) for name in channel_names[:5])
        if len(channel_names) > 5:
            shown += f", ... ({len(channel_names)} channels)"
        warnings.warn(
            f"The BrainVision channels {shown} declare the unit {unit!r}, which is not a voltage. Their values "
            "are being written to an NWB ElectricalSeries as though they were microvolts, so they will be wrong "
            "by whatever factor separates that unit from microvolts. Drop them with channels_to_skip if they "
            "are not neural data.",
            UserWarning,
            stacklevel=4,
        )
    return gains


def _parse_date(value: str, marker_path: Path) -> datetime | None:
    """Parse a New Segment date, ``YYYYMMDDhhmmss`` followed by six digits of microseconds."""
    value = value.strip()
    if not value or value in _PLACEHOLDER_DATES:
        return None
    if re.fullmatch(r"\d{20}", value):
        try:
            return datetime.strptime(value, "%Y%m%d%H%M%S%f")
        except ValueError:
            pass
    warnings.warn(
        f"The BrainVision marker file '{marker_path.name}' holds the New Segment date {value!r}, which is not a "
        "date in the format's YYYYMMDDhhmmssuuuuuu form; it is ignored.",
        UserWarning,
        stacklevel=5,
    )
    return None


def _parse_marker(entry: str, marker_path: Path, key: str) -> BrainVisionMarker | None:
    """
    Parse ``<type>,<description>,<position>,<size>,<channel>[,<date>]``.

    The format escapes commas in the type and description, but not every writer does. When the fields
    after the description do not parse, the numeric fields are found from the right and anything left
    over is taken to be part of the description.
    """
    fields = entry.split(",")

    def _numeric_tail(tail):
        position, size, channel = (tail + ["", ""])[:3]
        return int(position), int(size) if size.strip() else 0, int(channel) if channel.strip() else 0

    try:
        if len(fields) < 3:
            raise ValueError
        marker_type, description = fields[0], fields[1]
        position, size, channel = _numeric_tail(fields[2:5])
        date = fields[5] if len(fields) > 5 else ""
    except ValueError:
        # Unescaped commas in the description: the last three or four fields are the numeric tail.
        for tail_length in (4, 3):
            if len(fields) < tail_length + 2:
                continue
            tail = fields[-tail_length:]
            date = tail[3] if tail_length == 4 else ""
            if tail_length == 4 and date.strip() and not date.strip().isdigit():
                continue
            try:
                position, size, channel = _numeric_tail(tail[:3])
            except ValueError:
                continue
            marker_type, description = fields[0], ",".join(fields[1:-tail_length])
            break
        else:
            warnings.warn(
                f"Skipping the unreadable marker {key}={entry!r} in '{marker_path.name}'.",
                UserWarning,
                stacklevel=4,
            )
            return None

    marker_type = marker_type.strip().replace(r"\1", ",")
    return BrainVisionMarker(
        marker_type=marker_type,
        description=description.strip().replace(r"\1", ","),
        # Positions count from 1.
        sample_index=position - 1,
        size=size,
        channel_number=channel,
        date=_parse_date(date, marker_path=marker_path) if marker_type == _NEW_SEGMENT else None,
    )


def _read_markers(marker_path: Path) -> list[BrainVisionMarker]:
    marker_infos = _read_ini(marker_path).get("marker infos", {})

    def _marker_number(key):
        match = re.fullmatch(r"mk(\d+)", key)
        return int(match.group(1)) if match else None

    markers = []
    for key in sorted((key for key in marker_infos if _marker_number(key) is not None), key=_marker_number):
        marker = _parse_marker(marker_infos[key], marker_path=marker_path, key=f"Mk{_marker_number(key)}")
        if marker is not None:
            markers.append(marker)
    return markers


def _build_segments(
    markers: list[BrainVisionMarker], number_of_samples: int, sampling_frequency: float, marker_path: Path | None
) -> list[BrainVisionSegment]:
    """
    Split the samples at ``New Segment`` markers and place each run at its real time.

    A New Segment marker is written whenever recording resumes after a pause; the samples stay
    back-to-back in the file, and only the marker's date says how much time passed. A segment is placed
    at its own date relative to the first segment's when both are known, and directly after the
    previous segment otherwise.
    """
    boundaries = {0: None}
    for marker in markers:
        if marker.marker_type == _NEW_SEGMENT and 0 <= marker.sample_index < number_of_samples:
            if marker.sample_index not in boundaries or boundaries[marker.sample_index] is None:
                boundaries[marker.sample_index] = marker.date

    starts = sorted(boundaries)
    segments = []
    for index, first_sample in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else number_of_samples
        date = boundaries[first_sample]
        if index == 0:
            start_time = first_sample / sampling_frequency
        else:
            previous = segments[-1]
            contiguous = previous.start_time + previous.number_of_samples / sampling_frequency
            start_time = contiguous
            if date is not None and segments[0].date is not None:
                dated = segments[0].start_time + (date - segments[0].date).total_seconds()
                # A clock that steps backwards would make segments overlap on the timeline, which no
                # consumer can index; such a segment is kept in file order instead.
                if dated < contiguous - 0.5 / sampling_frequency:
                    warnings.warn(
                        f"In '{marker_path.name}', the New Segment at sample {first_sample + 1} is dated {date}, "
                        "earlier than the end of the segment before it. It is placed directly after that "
                        "segment instead.",
                        UserWarning,
                        stacklevel=3,
                    )
                else:
                    start_time = dated
        segments.append(
            BrainVisionSegment(
                first_sample=first_sample, number_of_samples=end - first_sample, start_time=start_time, date=date
            )
        )
    return segments


def read_brainvision_header(file_path) -> BrainVisionHeader:
    """
    Read a BrainVision ``.vhdr`` file together with its ``.vmrk`` markers, without loading the samples.

    Parameters
    ----------
    file_path : str or Path
        Path to the ``.vhdr`` header file.
    """
    header_path = Path(file_path)
    if header_path.suffix.lower() in (".eeg", ".vmrk"):
        raise ValueError(
            f"'{header_path.name}' is a BrainVision {header_path.suffix} file; pass the .vhdr header that "
            f"describes it ('{header_path.with_suffix('.vhdr').name}') instead."
        )
    sections = _read_ini(header_path)

    common = sections.get("common infos")
    if common is None:
        raise ValueError(f"'{header_path.name}' has no [Common Infos] section, so it is not a BrainVision header.")

    def _required(key):
        if key not in common:
            raise ValueError(f"The BrainVision header '{header_path.name}' has no {key!r} in [Common Infos].")
        return common[key]

    data_format = common.get("dataformat", "BINARY").upper()
    if data_format != "BINARY":
        raise NotImplementedError(
            f"'{header_path.name}' stores its samples as {data_format} text; only BINARY BrainVision data is supported."
        )
    data_type = common.get("datatype", "TIMEDOMAIN").upper()
    if data_type != "TIMEDOMAIN":
        raise NotImplementedError(f"'{header_path.name}' holds {data_type} data; only TIMEDOMAIN is supported.")
    segmentation = common.get("segmentationtype", "NOTSEGMENTED").upper()
    if segmentation != "NOTSEGMENTED":
        raise NotImplementedError(
            f"'{header_path.name}' holds data segmented by BrainVision Analyzer (SegmentationType={segmentation}); "
            "only continuous recordings are supported."
        )

    orientation = common.get("dataorientation", "MULTIPLEXED").upper()
    if orientation not in ("MULTIPLEXED", "VECTORIZED"):
        raise ValueError(f"'{header_path.name}' has the unknown DataOrientation {orientation!r}.")

    binary_format = sections.get("binary infos", {}).get("binaryformat", "").upper()
    if binary_format not in _BINARY_FORMAT_TO_DTYPE:
        raise NotImplementedError(
            f"'{header_path.name}' has the BinaryFormat {binary_format or '(none)'!r}; supported formats are "
            f"{', '.join(_BINARY_FORMAT_TO_DTYPE)}."
        )
    dtype = _BINARY_FORMAT_TO_DTYPE[binary_format]

    number_of_channels = int(_required("numberofchannels"))
    sampling_frequency = 1e6 / float(_required("samplinginterval"))

    data_path = _resolve_companion(header_path, common.get("datafile"), ".eeg", "data")
    if data_path is None:
        raise FileNotFoundError(
            f"The BrainVision data file for '{header_path.name}' was not found (named "
            f"{common.get('datafile')!r}, and no '{header_path.with_suffix('.eeg').name}' beside it)."
        )

    bytes_per_frame = number_of_channels * np.dtype(dtype).itemsize
    file_size = data_path.stat().st_size
    available_samples = file_size // bytes_per_frame
    declared_samples = int(common["datapoints"]) if common.get("datapoints", "").isdigit() else None
    if orientation == "VECTORIZED":
        # Channels are stored one after another, so the per-channel length must be known exactly or every
        # channel after the first is read from the wrong place.
        number_of_samples = declared_samples if declared_samples is not None else available_samples
        if number_of_samples * bytes_per_frame != file_size:
            raise ValueError(
                f"'{data_path.name}' holds {file_size} bytes, which is not {number_of_channels} channels of "
                f"{number_of_samples} {binary_format} samples stored channel after channel; the file is "
                "truncated or the header does not describe it."
            )
    else:
        number_of_samples = available_samples
        if file_size % bytes_per_frame or (declared_samples is not None and declared_samples > available_samples):
            warnings.warn(
                f"'{data_path.name}' ends partway through the data the header describes; reading the "
                f"{available_samples} complete samples it holds.",
                UserWarning,
                stacklevel=2,
            )

    names, references, resolutions, units = _parse_channels(
        sections.get("channel infos", {}), number_of_channels=number_of_channels, header_path=header_path
    )
    gains = _gains_to_microvolts(resolutions=resolutions, units=units, names=names)

    marker_path = _resolve_companion(header_path, common.get("markerfile"), ".vmrk", "marker")
    if marker_path is None and common.get("markerfile"):
        warnings.warn(
            f"The BrainVision marker file {common['markerfile']!r} named by '{header_path.name}' was not found; "
            "the recording is read without markers.",
            UserWarning,
            stacklevel=2,
        )
    markers = _read_markers(marker_path) if marker_path is not None else []
    segments = _build_segments(
        markers=markers,
        number_of_samples=number_of_samples,
        sampling_frequency=sampling_frequency,
        marker_path=marker_path,
    )

    return BrainVisionHeader(
        header_path=header_path,
        data_path=data_path,
        marker_path=marker_path,
        sampling_frequency=sampling_frequency,
        dtype=dtype,
        orientation=orientation,
        number_of_samples=number_of_samples,
        channel_names=names,
        references=references,
        resolutions=resolutions,
        units=units,
        gains_to_microvolts=gains,
        markers=markers,
        segments=segments,
    )


def memmap_samples(header: BrainVisionHeader) -> np.memmap:
    """Map the ``.eeg`` file as a ``(samples, channels)`` array, whichever orientation it is stored in."""
    if header.orientation == "MULTIPLEXED":
        return np.memmap(
            header.data_path, dtype=header.dtype, mode="r", shape=(header.number_of_samples, header.number_of_channels)
        )
    by_channel = np.memmap(
        header.data_path, dtype=header.dtype, mode="r", shape=(header.number_of_channels, header.number_of_samples)
    )
    return by_channel.T
