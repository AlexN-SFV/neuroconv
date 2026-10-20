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

from .._voltage_units import unit_to_microvolts

# The binary formats of the core specification, little-endian.
_BINARY_FORMAT_TO_DTYPE = {"INT_16": "<i2", "INT_32": "<i4", "IEEE_FLOAT_32": "<f4"}

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

    @property
    def is_new_segment(self) -> bool:
        """Whether the marker records a resumption of recording, rather than an event in the data."""
        return self.marker_type == _NEW_SEGMENT


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
    is_epoched: bool = False

    @property
    def number_of_channels(self) -> int:
        return len(self.channel_names)

    @property
    def recording_start(self) -> datetime | None:
        """
        Wall-clock time of the first sample, when the ``New Segment`` marker at the first sample states one.

        A date on a later segment is not carried back to the first, since the pauses before it are unknown.
        """
        return self.segments[0].date if self.segments else None


def _decode_text(raw: bytes) -> str:
    """
    Decode a header or marker file, as UTF-8 when it is valid UTF-8 and by its declared ``Codepage`` otherwise.

    Files written before the format adopted UTF-8 declare ``ANSI`` (Windows-1252) or nothing at all;
    AnyWave's exporter, for one, writes the micro sign as the single Latin-1 byte 0xB5 with no
    ``Codepage`` line. UTF-8 is tried first whatever the declaration says, because writers also declare
    ``ANSI`` over UTF-8 text, and because UTF-8 fails loudly on the lone high bytes of a single-byte
    encoding while Windows-1252 accepts almost any byte sequence and would turn UTF-8 into mojibake.
    Latin-1, which accepts every byte, is the last resort.
    """
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]

    # The key itself is ASCII in every encoding, so it can be found before the encoding is known.
    declared = re.search(rb"^\s*Codepage\s*=\s*([^\r\n;]+)", raw, re.IGNORECASE | re.MULTILINE)
    candidates = ["utf-8"]
    if declared:
        codepage = declared.group(1).decode("ascii", "replace").strip()
        candidates.append("cp1252" if codepage.upper() == "ANSI" else codepage)
    candidates.append("cp1252")

    for encoding in candidates:
        try:
            return raw.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("latin-1")


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
    return _parse_ini(_decode_text(file_path.read_bytes()))


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
    return unit.strip() or "µV"


def _parse_number(value: str, kind: type, header_path: Path, field: str):
    """Parse a numeric header field, naming the file and the field when it is not a number."""
    try:
        return kind(value.strip())
    except ValueError:
        description = "an integer" if kind is int else "a number"
        raise ValueError(
            f"The BrainVision header '{header_path.name}' gives {field} as {value!r}, which is not {description}."
        ) from None


def _parse_channels(
    channel_infos: dict[str, str], number_of_channels: int, header_path: Path, samples_are_floating_point: bool
):
    names, references, resolutions, units = [], [], [], []
    blank_resolutions, zero_resolutions = [], []
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
        resolutions.append(_parse_number(resolution, float, header_path, f"the resolution of Ch{index}"))
        if resolutions[-1] == 0:
            zero_resolutions.append(f"Ch{index}={entry}")
        units.append(_parse_unit(unit))

    # Analyzer leaves the resolution blank on every floating-point export, whose samples are already in the
    # declared unit; its INT_16 export of the same data (GIN's File_brainvision_3) confirms it. Blank on
    # integer samples is the unusual case.
    if blank_resolutions and not samples_are_floating_point:
        warnings.warn(
            f"The BrainVision header '{header_path.name}' gives no resolution for {len(blank_resolutions)} "
            f"channel(s) ({', '.join(repr(name) for name in blank_resolutions[:5])}); their samples are taken "
            "to be in the declared unit already (resolution 1).",
            UserWarning,
            stacklevel=4,
        )
    if zero_resolutions:
        warnings.warn(
            f"The BrainVision header '{header_path.name}' gives a resolution of 0 for {len(zero_resolutions)} "
            f"channel(s) ({', '.join(repr(line) for line in zero_resolutions[:5])}), so every sample of them "
            "scales to 0. A decimal comma, as in '0,1' for 0.1, is read as a field separator and produces this.",
            UserWarning,
            stacklevel=4,
        )
    return names, references, np.asarray(resolutions, dtype="float64"), units


def _gains_to_microvolts(resolutions: np.ndarray, units: list[str], names: list[str]) -> np.ndarray:
    gains = np.empty(len(units), dtype="float64")
    non_voltage: dict[str, list[str]] = {}
    for index, unit in enumerate(units):
        # A unit that is not a voltage is still read, as microvolts with a warning, since dropping the
        # channel is the caller's decision.
        factor = unit_to_microvolts(unit)
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
    markers: list[BrainVisionMarker],
    number_of_samples: int,
    sampling_frequency: float,
    marker_path: Path | None,
    is_epoched: bool = False,
) -> list[BrainVisionSegment]:
    """
    Split the samples at ``New Segment`` markers and place each run at its real time.

    A New Segment marker is written whenever recording resumes after a pause; the samples stay
    back-to-back in the file, and only the marker's date says how much time passed. A dated segment is
    placed at its date relative to the latest dated segment before it, and any other segment directly
    after the previous one. A date with no dated segment before it to measure from cannot be placed, so
    the pause before it is lost; that is warned about.

    In an epoched export (``is_epoched``) each segment is an epoch cut from the original recording, and
    epochs cut from overlapping windows legitimately overlap in time. Each is written as its own series,
    so a dated epoch is placed at its date even when that is before the previous epoch ends; only a date
    before the first epoch is refused.
    """
    boundaries = {0: None}
    for marker in markers:
        if marker.is_new_segment and 0 <= marker.sample_index < number_of_samples:
            if marker.sample_index not in boundaries or boundaries[marker.sample_index] is None:
                boundaries[marker.sample_index] = marker.date

    starts = sorted(boundaries)
    segments = []
    anchor = None  # the latest segment so far with a date, which later dates are measured from
    unanchored = []  # first samples of dated segments with no dated segment before them
    for index, first_sample in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else number_of_samples
        date = boundaries[first_sample]
        if index == 0:
            start_time = first_sample / sampling_frequency
        else:
            previous = segments[-1]
            contiguous = previous.start_time + previous.number_of_samples / sampling_frequency
            start_time = contiguous
            if date is not None and anchor is None:
                unanchored.append(first_sample)
            elif date is not None:
                dated = anchor.start_time + (date - anchor.date).total_seconds()
                # In a recording, a clock that steps backwards would make segments overlap on the timeline,
                # which no consumer can index; such a segment is kept in file order instead. Epochs may
                # overlap, but not start before the session does.
                earliest = segments[0].start_time if is_epoched else contiguous - 0.5 / sampling_frequency
                if dated < earliest:
                    limit = "the first epoch" if is_epoched else "the end of the segment before it"
                    warnings.warn(
                        f"In '{marker_path.name}', the New Segment at sample {first_sample + 1} is dated {date}, "
                        f"earlier than {limit}. It is placed directly after the segment before it instead.",
                        UserWarning,
                        stacklevel=3,
                    )
                else:
                    start_time = dated
        segment = BrainVisionSegment(
            first_sample=first_sample, number_of_samples=end - first_sample, start_time=start_time, date=date
        )
        segments.append(segment)
        # Epochs are dated on one clock, the original recording's, so all are measured from the first dated
        # one, and an epoch whose date was refused does not move the rest. A recording's clock can be reset
        # between pauses, so there each dated segment becomes the reference for the next.
        if date is not None and (anchor is None or not is_epoched):
            anchor = segment

    if unanchored:
        shown = ", ".join(str(first_sample + 1) for first_sample in unanchored[:5])
        warnings.warn(
            f"In '{marker_path.name}', the New Segment marker at the first sample has no date, but the one(s) at "
            f"sample(s) {shown} do. With nothing to measure those dates from, the length of the pause before "
            "each is unknown, so each is placed directly after the segment before it, and the recording has no "
            "start time.",
            UserWarning,
            stacklevel=3,
        )
    return segments


def _check_epochs(common: dict[str, str], markers: list[BrainVisionMarker], number_of_samples: int, header_path: Path):
    """
    Check that an epoched (``SegmentationType=MARKERBASED``) export is laid out as the segment model reads it.

    BrainVision Analyzer writes such an export as a flat binary of equal-length epochs, back to back, each
    opened by a dated ``New Segment`` marker; that is the structure :func:`_build_segments` reads. A file
    whose markers do not partition the samples that way is refused rather than split at the wrong places.
    """
    epoch_length = _parse_number(common.get("segmentdatapoints", ""), int, header_path, "SegmentDataPoints")
    if epoch_length < 1 or number_of_samples % epoch_length:
        raise NotImplementedError(
            f"'{header_path.name}' is an epoched export of {number_of_samples} samples, which is not a whole number "
            f"of epochs of SegmentDataPoints={epoch_length}."
        )
    expected = set(range(0, number_of_samples, epoch_length))
    found = {marker.sample_index for marker in markers if marker.is_new_segment}
    if found != expected:
        raise NotImplementedError(
            f"'{header_path.name}' is an epoched export of {len(expected)} epochs of {epoch_length} samples, but "
            f"its New Segment markers do not open exactly those epochs ({len(found)} markers, "
            f"{len(found - expected)} not at an epoch start, {len(expected - found)} epochs without one)."
        )


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
    if segmentation not in ("NOTSEGMENTED", "MARKERBASED"):
        raise NotImplementedError(
            f"'{header_path.name}' holds data segmented by BrainVision Analyzer with SegmentationType={segmentation}; "
            "only continuous recordings and MARKERBASED epochs are supported."
        )
    if common.get("averaged", "NO").upper() == "YES":
        raise NotImplementedError(
            f"'{header_path.name}' holds an average over epochs (Averaged=YES), which has no time in the "
            "recording; only continuous recordings and individual epochs are supported."
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

    number_of_channels = _parse_number(_required("numberofchannels"), int, header_path, "NumberOfChannels")
    if number_of_channels < 1:
        raise ValueError(
            f"The BrainVision header '{header_path.name}' gives NumberOfChannels={number_of_channels}; a recording "
            "needs at least one channel."
        )
    sampling_interval = _parse_number(_required("samplinginterval"), float, header_path, "SamplingInterval")
    if not sampling_interval > 0:
        raise ValueError(
            f"The BrainVision header '{header_path.name}' gives SamplingInterval={sampling_interval}; the interval "
            "between samples, in microseconds, must be positive."
        )
    sampling_frequency = 1e6 / sampling_interval

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
        sections.get("channel infos", {}),
        number_of_channels=number_of_channels,
        header_path=header_path,
        samples_are_floating_point=np.dtype(dtype).kind == "f",
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
    is_epoched = segmentation == "MARKERBASED"
    if is_epoched:
        _check_epochs(common=common, markers=markers, number_of_samples=number_of_samples, header_path=header_path)
    segments = _build_segments(
        markers=markers,
        number_of_samples=number_of_samples,
        sampling_frequency=sampling_frequency,
        marker_path=marker_path,
        is_epoched=is_epoched,
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
        is_epoched=is_epoched,
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
