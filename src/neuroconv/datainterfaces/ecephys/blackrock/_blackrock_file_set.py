"""
Where the samples and events of a Blackrock file set (``.nsX`` and ``.nev``) go on the NWB timeline.

neo's ``BlackrockRawIO`` parses the headers and memory-maps the samples. This module decides how they are
split into runs and where each run and each ``.nev`` packet is placed, because what neo and SpikeInterface
do with the clock values cannot be written as they are:

* **t = 0 is the file set's earliest first sample.** ``TimeOrigin`` is the UTC wall time the recording was
  started, the same to the millisecond in every ``.nsX`` file of a set, and BIDS ``events.tsv`` onsets count
  from a data file's first sample. The clock, though, starts at the recording only in older files: a FileSpec
  3.0 file's first sample is at 893.7 s of the Neural Signal Processor's clock, and a PTP (Gemini) file's at
  its Unix-epoch nanoseconds (1.69e9 s), which are not even UTC (24 to 118 days off ``TimeOrigin`` in the
  test files). The streams of one set do not start on the same tick (the GIN ``.ns4`` and ``.ns6`` of one
  recording are 0.134 ms apart), so all of them, and the ``.nev``, share the earliest one as their origin;
  a stream that starts later starts that much after 0.
* **A clock reset does not overlap.** A ResetCorrect file restarts its clock near zero mid-file. The real
  start of what follows is unknown, but it cannot be before the end of what came before, so each run after a
  reset is placed at that end plus its own clock value since the reset, and flagged.
* **PTP packets are put in timestamp order, then split where a sample is missing.** The test files labelled
  "missing samples" miss none: sorted by timestamp, their spacing is uniform, but a block of packets is
  stored late. neo's ``gap_tolerance_ms`` keeps file order and places those samples up to 2 ms late, over
  the next segment. Here every gap of more than 1.5 sample periods starts a new run, so no tolerance needs
  choosing.

Each run is written at the nominal sampling rate from its first sample. ``.nev`` events and spikes are
placed on the same timeline: linearly on the clock for timestamped blocks, whose samples are evenly spaced
by definition, and through the samples they fall between for PTP, whose per-sample timestamps jitter and can
drift from the nominal rate, so an event stays on the sample it was recorded at.
"""

import datetime
import re
import struct
import warnings
import weakref
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .._channel_ids import _stacklevel_outside_neuroconv

_NSX_SUFFIX = re.compile(r"\.ns([1-6])$")
_FILE_DIALOG_VERSION = re.compile(r"File Dialog v(\d+)\.(\d+)")

# A gap of more than this many sample periods between consecutive PTP timestamps starts a new run. The
# timestamps jitter by under 1% of a period, and one missing sample makes a gap of 2.
_PTP_GAP_IN_PERIODS = 1.5

# Bits of a NonNeural packet's insertion-reason byte.
_REASON_DIGITAL = 1 << 0
_REASON_SERIAL = 1 << 7
_REASON_PERIODIC = 1 << 6

_COMMENT_ENCODINGS = {0: "latin-1", 1: "utf-16-le", 255: "latin-1"}
_CONTROL_CHARACTER = re.compile("[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]")


@dataclass
class BlackrockRun:
    """An uninterrupted run of samples: the arrays holding them, in order, and where the run starts."""

    pieces: list = field(default_factory=list)  # (array of shape (samples, channels), row index or None)
    number_of_samples: int = 0
    start_time: float = 0.0
    first_tick: int = 0
    after_clock_reset: bool = False


@dataclass
class _ClockEpoch:
    """A stretch of the clock between resets: a tick ``t`` of it is at ``base + (t - reference) / resolution``."""

    base: float
    reference: int


class BlackrockTimeline:
    """Maps clock values of a file set to seconds on the NWB timeline of one ``.nsX`` stream."""

    def __init__(self, *, resolution, sampling_frequency, epochs, runs, sorted_ticks=None):
        self.resolution = float(resolution)
        self.sampling_frequency = float(sampling_frequency)
        self.epochs = epochs
        self.runs = runs
        # The furthest a PTP run's last timestamp lies from where the nominal rate puts its sample, in seconds.
        self.largest_drift = 0.0
        # PTP only: every sample's timestamp in time order, left on disk where the file holds them in order.
        self._sorted_ticks = sorted_ticks
        if sorted_ticks is not None:
            counts = [run.number_of_samples for run in runs]
            self._run_firsts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
            self._run_starts = np.array([run.start_time for run in runs])

    @property
    def is_ptp(self) -> bool:
        return self._sorted_ticks is not None

    def to_seconds(self, ticks: np.ndarray, resolution: float, epochs: np.ndarray | None = None) -> np.ndarray:
        """
        Seconds on the timeline of the clock values ``ticks``, counted at ``resolution`` per second.

        ``epochs`` gives the clock reset each value falls after (0 before any); values in an epoch the stream
        does not have are returned as NaN.
        """
        ticks = np.asarray(ticks)
        if float(resolution) == self.resolution:
            ticks = ticks.astype(np.int64)
        else:
            ticks = ticks.astype(np.float64) * (self.resolution / float(resolution))
        if self.is_ptp:
            return self._ptp_seconds(ticks)

        epochs = np.zeros(len(ticks), dtype=int) if epochs is None else np.asarray(epochs, dtype=int)
        seconds = np.full(len(ticks), np.nan)
        for index, epoch in enumerate(self.epochs):
            mask = epochs == index
            seconds[mask] = epoch.base + (ticks[mask] - epoch.reference) / self.resolution
        return seconds

    def _ptp_seconds(self, ticks: np.ndarray) -> np.ndarray:
        # Each value is placed from the last sample stamped at or before it, so it stays on that sample however
        # far the per-sample timestamps have drifted from the nominal rate the series is written at.
        sorted_ticks = self._sorted_ticks
        positions = _searchsorted_right(sorted_ticks, ticks) - 1
        before_first = positions < 0
        positions = np.clip(positions, 0, len(sorted_ticks) - 1)
        runs = np.searchsorted(self._run_firsts, positions, side="right") - 1
        sample_seconds = self._run_starts[runs] + (positions - self._run_firsts[runs]) / self.sampling_frequency
        seconds = sample_seconds + (ticks - np.asarray(sorted_ticks[positions], dtype=np.int64)) / self.resolution
        # Before the first sample there is nothing to anchor to but the clock itself, from the set's origin.
        seconds[before_first] = (ticks[before_first] - self.epochs[0].reference) / self.resolution
        return seconds


# PTP timestamps are scanned and searched this many at a time, so the eight bytes per sample of a long recording
# stored in order (a day at 30 kHz is 21 GB of them) are never all in memory at once. A file with packets out of
# order is the exception: putting them back in order holds every timestamp and the order, 16 bytes per sample,
# which neo's own parse of the file, about 25 bytes per sample, already exceeds.
_CHUNK = 1 << 22


def _searchsorted_right(sorted_ticks, values: np.ndarray) -> np.ndarray:
    """``np.searchsorted(sorted_ticks, values, side="right")`` reading only small parts of ``sorted_ticks``."""
    if isinstance(sorted_ticks, np.ndarray) and sorted_ticks.flags.c_contiguous:
        return np.searchsorted(sorted_ticks, values, side="right")
    # np.searchsorted copies a strided array whole; find each value's chunk on every _CHUNK-th tick instead.
    coarse = np.asarray(sorted_ticks[::_CHUNK], dtype=np.int64)
    chunks = np.searchsorted(coarse, values, side="right") - 1
    positions = np.zeros(len(values), dtype=np.int64)
    for chunk in np.unique(chunks):
        mask = chunks == chunk
        if chunk < 0:
            continue
        first = int(chunk) * _CHUNK
        part = np.asarray(sorted_ticks[first : first + _CHUNK], dtype=np.int64)
        positions[mask] = first + np.searchsorted(part, values[mask], side="right")
    return positions


@dataclass
class BlackrockStream:
    """One ``.nsX`` file: its channels, its runs on the timeline and its header times."""

    path: Path
    nsx_number: int
    spec: str
    sampling_frequency: float
    channel_ids: list
    channel_labels: list
    units: list
    gains: np.ndarray
    offsets: np.ndarray
    timeline: BlackrockTimeline
    time_origin: datetime.datetime | None
    comment: str
    out_of_order_packets: int = 0
    warned: bool = False

    @property
    def runs(self) -> list[BlackrockRun]:
        return self.timeline.runs


@dataclass
class BlackrockEvents:
    """The ``.nev`` packets that are not spikes or headers: digital and serial input words, and comments."""

    ticks: np.ndarray
    epochs: np.ndarray
    kinds: list
    labels: list
    values: np.ndarray
    packet_insertion_reasons: np.ndarray


@dataclass
class BlackrockNev:
    """What neuroconv reads from a ``.nev`` file."""

    path: Path
    spec: str
    resolution: float
    sample_resolution: float
    time_origin: datetime.datetime
    creating_application: str
    digitization_factors: dict
    events: BlackrockEvents
    spike_ticks: np.ndarray
    spike_epochs: np.ndarray
    spike_electrodes: np.ndarray
    spike_units: np.ndarray
    unit_keys: list  # (electrode id, unit class number), in neo's order

    @property
    def is_ptp(self) -> bool:
        return self.resolution == 1e9

    def time_origin_may_be_local(self) -> bool:
        """
        Whether the application that wrote this ``.nev`` is one seen writing local time to it, not UTC.

        ``FileSpec2.3001.nev`` (File Dialog v6.04) says 13:23, seven hours behind the 20:23 UTC of its own
        ``.ns5``; from v6.05 on, the test files' two headers agree within milliseconds.
        """
        match = _FILE_DIALOG_VERSION.search(self.creating_application)
        return match is None or (int(match.group(1)), int(match.group(2))) < (6, 5)


def nsx_number(path: Path) -> int | None:
    """The stream number of an ``.nsX`` path (5 for ``.ns5``), or None for any other suffix."""
    match = _NSX_SUFFIX.search(Path(path).name)
    return int(match.group(1)) if match else None


def file_set_base(path: Path) -> Path:
    """The path without its ``.nsX`` or ``.nev`` suffix, keeping any dot in the name itself."""
    path = Path(path)
    if nsx_number(path) is not None or path.suffix == ".nev":
        return path.with_name(path.name[: -len(path.suffix)])
    return path


def available_nsx_files(base: Path) -> list[Path]:
    """The ``.ns1`` to ``.ns6`` files that exist for a file set's base path."""
    return [Path(f"{base}.ns{number}") for number in range(1, 7) if Path(f"{base}.ns{number}").is_file()]


def _neo_reader(**kwargs):
    from neo.rawio import BlackrockRawIO

    reader = BlackrockRawIO(**kwargs)
    with warnings.catch_warnings():
        # neo's own warning for a FileSpec 2.1 stream read without its .nev; the scaling is taken from the .nev
        # here instead, and a stream without one is refused before this point.
        warnings.filterwarnings("ignore", message="Cannot rescale to voltage")
        reader.parse_header()
    return reader


def _time_origin(header) -> datetime.datetime:
    return datetime.datetime(
        int(header["year"]),
        int(header["month"]),
        int(header["day"]),
        int(header["hour"]),
        int(header["minute"]),
        int(header["second"]),
        int(header["millisecond"]) * 1000,
        tzinfo=datetime.timezone.utc,
    )


def read_nev(nev_path: Path) -> BlackrockNev:
    """Read a ``.nev`` file's header, events and spike times, through neo, without touching any ``.nsX``."""
    nev_path = Path(nev_path)
    base = file_set_base(nev_path)
    # neo pairs the .nev with every .nsX of the same base name, and with nsx_to_load=None would parse all of
    # them. Pointing nsx_override at a name no file has makes it read the .nev alone.
    with warnings.catch_warnings():
        # neo reports each backward step in the .nev's timestamps as an undocumented segment; the resets are
        # reported against the .nsX here, and in PTP files the steps are packets stored out of order.
        warnings.filterwarnings("ignore", message="Detected .* undocumented segments")
        reader = _neo_reader(filename=str(base), nsx_override=f"{base}.neuroconv-no-nsx", nsx_to_load=None)

    header = reader._nev_basic_header
    spec = reader._nev_spec
    resolution = float(header["timestamp_resolution"])
    factors = {}
    if b"NEUEVWAV" in reader._nev_ext_header:
        factors = {int(key): float(value) for key, value in reader._nev_params("digitization_factor").items()}

    spikes, spike_epochs = reader.nev_data["Spikes"]
    events = _read_events(reader, spec)
    if resolution == 1e9:
        # The PTP clock does not reset; a backward step between packets is a packet stored out of order, which
        # neo counts as a reset.
        spike_epochs = np.zeros(len(spikes), dtype=int)
        events.epochs = np.zeros(len(events.ticks), dtype=int)

    return BlackrockNev(
        path=nev_path,
        spec=spec,
        resolution=resolution,
        sample_resolution=float(header["sample_resolution"]),
        time_origin=_time_origin(header),
        creating_application=bytes(header["application_to_create_file"]).split(b"\x00")[0].decode("latin-1"),
        digitization_factors=factors,
        events=events,
        spike_ticks=np.asarray(spikes["timestamp"]),
        spike_epochs=np.asarray(spike_epochs, dtype=int),
        spike_electrodes=np.asarray(spikes["packet_id"], dtype=int),
        spike_units=np.asarray(spikes["unit_class_nb"], dtype=int),
        unit_keys=[(int(channel_id), int(unit_id)) for channel_id, unit_id in reader.internal_unit_ids],
    )


def _decode_comment(raw: bytes, char_set: int) -> str:
    encoding = _COMMENT_ENCODINGS.get(int(char_set), "latin-1")
    if encoding == "utf-16-le":
        # numpy strips the trailing NUL bytes of a fixed-width field, which can take the high byte of the last
        # character with them.
        raw = raw + b"\x00" * (len(raw) % 2)
    text = raw.decode(encoding, errors="replace")
    # The field is not always terminated: after an eight-character comment FileSpec2.3001.nev holds the packet's
    # own timestamp ('Comment2\x03\xda\x0b' is 776707 ticks), so the text ends at the first control character.
    return _CONTROL_CHARACTER.split(text, 1)[0]


def _read_events(reader, spec: str) -> BlackrockEvents:
    ticks, epochs, kinds, labels, values, reasons = [], [], [], [], [], []

    packets, packet_epochs = reader.nev_data["NonNeural"]
    for packet, epoch in zip(packets, packet_epochs):
        reason = int(packet["packet_insertion_reason"])
        word = int(packet["digital_input"])
        label = str(word)
        if reason & _REASON_DIGITAL:
            kind = "serial_input" if reason & _REASON_SERIAL else "digital_input"
        elif spec in ("2.1", "2.2") and reason & 0b111110:
            # FileSpec 2.1 and 2.2 also insert a packet when one of five analog inputs crosses its threshold.
            kind = "analog_input"
            label = ", ".join(
                f"analog_input_channel_{channel}={int(packet[f'analog_input_channel_{channel}'])}"
                for channel in range(1, 6)
                if reason & (1 << channel)
            )
        elif reason & _REASON_PERIODIC:
            kind = "periodic_sampling"
        else:
            kind = "unknown"
        ticks.append(int(packet["timestamp"]))
        epochs.append(int(epoch))
        kinds.append(kind)
        labels.append(label)
        values.append(word)
        reasons.append(reason)

    if "Comments" in reader.nev_data:
        comments, comment_epochs = reader.nev_data["Comments"]
        for comment, epoch in zip(comments, comment_epochs):
            ticks.append(int(comment["timestamp"]))
            epochs.append(int(epoch))
            # Character set 255 marks a NeuroMotive tracking event stored as a comment ("ROI name:number:event").
            kinds.append("tracking_event" if int(comment["char_set"]) == 255 else "comment")
            labels.append(_decode_comment(bytes(comment["comment"]), comment["char_set"]))
            values.append(-1)
            reasons.append(-1)

    return BlackrockEvents(
        ticks=np.asarray(ticks, dtype=np.uint64),
        epochs=np.asarray(epochs, dtype=int),
        kinds=kinds,
        labels=labels,
        values=np.asarray(values, dtype=np.int64),
        packet_insertion_reasons=np.asarray(reasons, dtype=np.int64),
    )


# Streams and .nev files already read, so a recording and a sorting interface over one set in a ConverterPipe
# parse each file once, and warn about it once. Weak values: an entry goes with the last extractor holding it.
_STREAMS = weakref.WeakValueDictionary()
_NEVS = weakref.WeakValueDictionary()


def _file_key(path: Path | None):
    if path is None:
        return None
    stat = Path(path).stat()
    return str(Path(path).resolve()), stat.st_size, stat.st_mtime_ns


def read_nev_cached(nev_path: Path) -> BlackrockNev:
    """:func:`read_nev`, reusing what was read from an unchanged file."""
    key = _file_key(nev_path)
    nev = _NEVS.get(key)
    if nev is None:
        nev = _NEVS[key] = read_nev(nev_path)
    return nev


def read_stream(nsx_path: Path, nev: BlackrockNev | None = None) -> BlackrockStream:
    """
    Read one ``.nsX`` file's channels and lay its samples out in runs, reusing what was read from an unchanged file.

    ``nev`` is needed for a FileSpec 2.1 stream, whose own header carries neither the scaling of its samples
    nor a start time; both are in the ``.nev``.
    """
    nsx_path = Path(nsx_path)
    if nsx_number(nsx_path) is None:
        raise ValueError(f"'{nsx_path}' is not a Blackrock continuous data file: its suffix is not .ns1 to .ns6.")
    key = (_file_key(nsx_path), None if nev is None else _file_key(nev.path))
    stream = _STREAMS.get(key)
    if stream is None:
        stream = _STREAMS[key] = _read_stream(nsx_path, nev)
    return stream


def _read_stream(nsx_path: Path, nev: BlackrockNev | None) -> BlackrockStream:
    number = nsx_number(nsx_path)
    reader = _neo_reader(
        filename=str(file_set_base(nsx_path)),
        nsx_to_load=number,
        load_nev=False,
        # PTP streams are split into runs here; an infinite tolerance keeps neo from splitting, or refusing, them.
        gap_tolerance_ms=float("inf"),
    )
    spec = reader._nsx_spec[number]
    basic = reader._nsx_basic_header[number]
    sampling_frequency = reader._nsx_sampling_frequency[number]
    channels = reader.header["signal_channels"]

    channel_ids = [str(channel_id) for channel_id in channels["id"]]
    labels = [str(name) for name in channels["name"]]
    units = [str(unit) for unit in channels["units"]]
    gains = np.asarray(channels["gain"], dtype=np.float64)
    offsets = np.asarray(channels["offset"], dtype=np.float64)

    if spec == "2.1":
        if nev is None:
            raise ValueError(
                f"'{nsx_path.name}' is a FileSpec 2.1 file, whose header gives no scaling for its samples and no "
                "start time: both are in the .nev file recorded with it. Put that file beside it with the same "
                "base name, or pass its path as nev_file_path."
            )
        gains, units = _gains_from_nev(channel_ids=channel_ids, nev=nev, nsx_path=nsx_path)
        offsets = np.zeros(len(channel_ids))
        blocks = [(_full_v21_data(reader, number, nsx_path), None)]
        time_origin = None
    else:
        blocks = [
            (reader.nsx_datas[number][segment], reader._nsx_data_header[number][segment]["timestamp"])
            for segment in sorted(reader.nsx_datas[number])
        ]
        time_origin = _time_origin(basic)

    resolution = float(basic["timestamp_resolution"]) if "timestamp_resolution" in basic.dtype.names else 30_000.0
    ticks_per_sample = resolution / sampling_frequency
    is_ptp = len(blocks) == 1 and isinstance(blocks[0][1], np.ndarray) and blocks[0][1].size == len(blocks[0][0])

    # The origin of every stream of the set; a FileSpec 2.1 clock is not stored, and starts with the recording.
    origin = None if spec == "2.1" else set_origin(nsx_path, resolution)
    out_of_order = 0
    if is_ptp:
        timeline, out_of_order = _ptp_timeline(blocks[0], resolution, sampling_frequency, ticks_per_sample, origin)
    else:
        timeline = _block_timeline(blocks, resolution, sampling_frequency, ticks_per_sample, origin)

    comment = bytes(basic["comment"]).split(b"\x00")[0].decode("latin-1") if "comment" in basic.dtype.names else ""
    return BlackrockStream(
        path=nsx_path,
        nsx_number=number,
        spec=spec,
        sampling_frequency=sampling_frequency,
        channel_ids=channel_ids,
        channel_labels=labels,
        units=units,
        gains=gains,
        offsets=offsets,
        timeline=timeline,
        time_origin=time_origin,
        comment=comment.strip(),
        out_of_order_packets=out_of_order,
    )


def _full_v21_data(reader, number: int, nsx_path: Path) -> np.ndarray:
    # neo leaves out the last sample of a FileSpec 2.1 file (its count subtracts one); NPMK's openNSx, like the
    # file size, has them all.
    channel_count = int(reader._nsx_basic_header[number]["channel_count"])
    header_bytes = (
        reader._nsx_basic_header[number].dtype.itemsize + reader._nsx_ext_header[number].dtype.itemsize * channel_count
    )
    number_of_samples = (nsx_path.stat().st_size - header_bytes) // (2 * channel_count)
    return np.memmap(nsx_path, dtype="<i2", mode="r", offset=header_bytes, shape=(number_of_samples, channel_count))


def _gains_from_nev(channel_ids: list, nev: BlackrockNev, nsx_path: Path):
    """FileSpec 2.1 gains: the .nev's digitization factor, in nanovolts per count, for each electrode."""
    gains, missing = [], []
    for channel_id in channel_ids:
        factor = nev.digitization_factors.get(int(channel_id))
        if factor is None:
            missing.append(channel_id)
            factor = float("nan")
        elif factor == 21516:
            # A Cerebus firmware bug overflows the analog inputs' factor; NPMK's openNEV substitutes this value.
            factor = 152592.547
        gains.append(factor / 1000.0)
    if missing:
        raise ValueError(
            f"The .nev file '{nev.path.name}' gives no digitization factor for electrodes {missing} of "
            f"'{nsx_path.name}', so their samples cannot be scaled."
        )
    return np.asarray(gains), ["uV"] * len(channel_ids)


def first_sample_tick(nsx_path: Path) -> tuple[int, float] | None:
    """
    The clock value of a ``.nsX`` file's first sample and the clock's resolution, read from its headers alone.

    Blocks shorter than two samples are skipped, as neo drops them. A PTP file's earliest timestamp is taken from
    its first chunk of packets, since packets stored out of order are out of order only locally. ``None`` for a
    file whose data cannot be read this way.
    """
    nsx_path = Path(nsx_path)
    size = nsx_path.stat().st_size
    with open(nsx_path, "rb") as file:
        basic = file.read(314)
        if basic[:8] == b"NEURALSG":
            return 0, 30_000.0
        if len(basic) < 314:
            return None
        major = basic[8]
        (bytes_in_headers,) = struct.unpack_from("<I", basic, 10)
        (resolution,) = struct.unpack_from("<I", basic, 290)
        (channel_count,) = struct.unpack_from("<I", basic, 310)
        if resolution == 1_000_000_000:
            packet = np.dtype(
                [("reserved", "u1"), ("timestamp", "<u8"), ("count", "<u4"), ("samples", "<i2", (channel_count,))]
            )
            number_of_packets = min((size - bytes_in_headers) // packet.itemsize, _CHUNK)
            if number_of_packets > 0:
                packets = np.memmap(
                    nsx_path, dtype=packet, mode="r", offset=bytes_in_headers, shape=(number_of_packets,)
                )
                if np.all(packets["count"] == 1):
                    return int(packets["timestamp"].min()), float(resolution)
        header = struct.Struct("<BQI" if major >= 3 else "<BII")
        offset = bytes_in_headers
        while offset + header.size <= size:
            file.seek(offset)
            flag, tick, number_of_samples = header.unpack(file.read(header.size))
            if flag != 1:
                return None
            if number_of_samples >= 2:
                return int(tick), float(resolution)
            offset += header.size + number_of_samples * channel_count * 2
    return None


def set_origin(nsx_path: Path, resolution: float) -> int | None:
    """
    The earliest first sample among the ``.nsX`` files of ``nsx_path``'s set, in ticks of ``resolution``.

    Every stream of a set, and its ``.nev``, is placed from this one origin, so they line up in one NWB file.
    """
    candidates = {Path(nsx_path), *available_nsx_files(file_set_base(nsx_path))}
    firsts = [first for first in map(first_sample_tick, sorted(candidates)) if first is not None]
    if not firsts:
        return None
    tick, tick_resolution = min(firsts, key=lambda first: first[0] / first[1])
    return tick if tick_resolution == resolution else int(round(tick * resolution / tick_resolution))


def _block_timeline(blocks, resolution, sampling_frequency, ticks_per_sample, origin=None) -> BlackrockTimeline:
    """
    Runs of a stream whose data blocks each carry the clock value of their first sample (or none, FileSpec 2.1).

    A block that starts where the last one ended continues its run; one that starts later begins a new run
    (a pause); one that starts before the last one ended follows a clock reset. Before any reset, a tick is placed
    from ``origin``, the set's earliest first sample (this stream's own when ``None``).
    """
    runs, epochs = [], []
    previous_end = None
    for data, timestamp in blocks:
        tick = 0 if timestamp is None else int(timestamp)
        number_of_samples = int(data.shape[0])
        if not runs:
            reference = tick if origin is None else min(origin, tick)
            epochs.append(_ClockEpoch(base=0.0, reference=reference))
            runs.append(BlackrockRun(first_tick=tick, start_time=(tick - reference) / resolution))
        elif tick < previous_end - ticks_per_sample / 2:
            last = runs[-1]
            epochs.append(_ClockEpoch(base=last.start_time + last.number_of_samples / sampling_frequency, reference=0))
            runs.append(
                BlackrockRun(
                    first_tick=tick,
                    start_time=epochs[-1].base + tick / resolution,
                    after_clock_reset=True,
                )
            )
        elif tick > previous_end + ticks_per_sample / 2:
            epoch = epochs[-1]
            runs.append(
                BlackrockRun(
                    first_tick=tick,
                    start_time=epoch.base + (tick - epoch.reference) / resolution,
                )
            )
        run = runs[-1]
        run.pieces.append((data, None))
        run.number_of_samples += number_of_samples
        previous_end = tick + number_of_samples * ticks_per_sample
    return BlackrockTimeline(
        resolution=resolution, sampling_frequency=sampling_frequency, epochs=epochs, runs=runs, sorted_ticks=None
    )


def _ptp_timeline(block, resolution, sampling_frequency, ticks_per_sample, origin=None):
    """
    Runs of a PTP stream, one packet per sample, each with its own timestamp in nanoseconds.

    Ticks are placed from ``origin``, the set's earliest first sample (this stream's own when ``None``).
    """
    data, timestamps = block
    # A view, not a copy: the timestamps stay in the memory map, read a chunk at a time.
    ticks = timestamps.view(np.int64) if timestamps.dtype.itemsize == 8 else np.asarray(timestamps, dtype=np.int64)
    gap = _PTP_GAP_IN_PERIODS * ticks_per_sample
    order = None
    gaps, backward = _scan_ptp_ticks(ticks, gap)
    if backward:
        # Packets out of order are put back in order, which needs every timestamp in memory.
        order = np.argsort(np.asarray(ticks), kind="stable")
        ticks = np.asarray(ticks)[order]
        gaps, _ = _scan_ptp_ticks(ticks, gap)
    out_of_order = 0 if order is None else _count_moved(order)

    starts = np.concatenate([[0], gaps]).astype(np.int64)
    stops = np.concatenate([starts[1:], [len(ticks)]])
    first_tick = int(ticks[0]) if origin is None else min(origin, int(ticks[0]))
    runs = []
    for start, stop in zip(starts, stops):
        if order is None:
            rows = slice(int(start), int(stop))
        else:
            rows = order[start:stop]
            # Most runs are untouched by the reordering; those are read as a slice.
            if np.array_equal(rows, np.arange(rows[0], rows[0] + len(rows))):
                rows = slice(int(rows[0]), int(rows[0]) + len(rows))
        runs.append(
            BlackrockRun(
                pieces=[(data, rows)],
                number_of_samples=int(stop - start),
                start_time=(int(ticks[start]) - first_tick) / resolution,
                first_tick=int(ticks[start]),
            )
        )
    timeline = BlackrockTimeline(
        resolution=resolution,
        sampling_frequency=sampling_frequency,
        epochs=[_ClockEpoch(base=0.0, reference=first_tick)],
        runs=runs,
        sorted_ticks=ticks,
    )
    timeline.largest_drift = _largest_ptp_drift(ticks, starts, stops, ticks_per_sample) / resolution
    return timeline, out_of_order


def _count_moved(order: np.ndarray) -> int:
    """How many packets the reordering moves, compared a chunk at a time rather than against a whole arange."""
    return sum(
        int(np.count_nonzero(order[first : first + _CHUNK] != np.arange(first, min(first + _CHUNK, len(order)))))
        for first in range(0, len(order), _CHUNK)
    )


def _scan_ptp_ticks(ticks, gap: float) -> tuple[np.ndarray, bool]:
    """Where a new run starts (the sample after a step of more than ``gap`` ticks), and whether any step is back."""
    gaps, backward = [], False
    for first in range(0, len(ticks) - 1, _CHUNK):
        # Each chunk overlaps the next by one sample, so the step between them is seen.
        steps = np.diff(np.asarray(ticks[first : first + _CHUNK + 1], dtype=np.int64))
        backward = backward or bool((steps < 0).any())
        gaps.append(first + np.flatnonzero(steps > gap) + 1)
    return (np.concatenate(gaps) if gaps else np.zeros(0, dtype=np.int64)), backward


def _largest_ptp_drift(ticks, starts, stops, ticks_per_sample) -> float:
    """The furthest any run's last timestamp lies from where the nominal rate puts its last sample, in ticks."""
    largest = 0.0
    for start, stop in zip(starts, stops):
        nominal = int(ticks[start]) + (stop - 1 - start) * ticks_per_sample
        largest = max(largest, abs(float(ticks[stop - 1]) - nominal))
    return largest


def warn_about_stream(stream: BlackrockStream) -> None:
    """Say what was done to place a stream's samples, where it was more than reading them in order, once per stream."""
    if stream.warned:
        return
    stream.warned = True
    name = stream.path.name
    resets = sum(run.after_clock_reset for run in stream.runs)
    if resets:
        warnings.warn(
            f"The clock of '{name}' was reset {resets} time(s) during the recording. The time of what was recorded "
            "after a reset is not in the file; each run after one is placed where the run before it ends, plus its "
            "own clock value since the reset, which is the earliest it can have started. Those runs are marked in "
            "the runs table's 'after_clock_reset' column.",
            UserWarning,
            stacklevel=_stacklevel_outside_neuroconv(),
        )
    if stream.out_of_order_packets:
        warnings.warn(
            f"{stream.out_of_order_packets} of the packets of '{name}' are stored out of timestamp order. They are "
            "read in timestamp order, so each sample is at its own timestamp.",
            UserWarning,
            stacklevel=_stacklevel_outside_neuroconv(),
        )
    drift = stream.timeline.largest_drift
    if drift * stream.sampling_frequency > 1:
        warnings.warn(
            f"The per-sample timestamps of '{name}' drift up to {drift * 1e3:.3f} ms from the nominal sampling rate "
            f"of {stream.sampling_frequency:g} Hz within a run. The samples are written at the nominal rate; .nev "
            "events and spikes are placed on the samples they were recorded with, so they stay aligned with the data.",
            UserWarning,
            stacklevel=_stacklevel_outside_neuroconv(),
        )
