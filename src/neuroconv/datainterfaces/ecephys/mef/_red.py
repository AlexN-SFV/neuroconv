"""
Decoder for RED (Range Encoded Differences), the block compression of MEF 2 and MEF 3.0 time series.

A RED block stores the differences between consecutive samples as one byte each, range coded against a
table of 256 symbol counts kept in the block header. A difference that does not fit in a byte is written as
a keysample instead: the reserved symbol -128 followed by the sample's full value. The first sample of every
block is a keysample, so each block decodes on its own.

The two formats differ only in where things sit, which :class:`REDLayout` carries: the block header is 304
bytes in MEF 3.0 and 287 in MEF 2, MEF 2 puts one byte between the header and the coded data, and a
keysample is 4 bytes in MEF 3.0 and 3 in MEF 2. The format readers parse their own block headers and pass
the per-block fields in. MEF 2.0 and 2.1 code their blocks identically.

Decoding follows ``RED_decode`` in meflib (``meflib.c``, the reader pymef wraps) step for step, including the
two transforms a writer can apply before coding to compress further: dividing a block by a scale factor
(``RED_unscale`` multiplies it back) and subtracting a fitted line (``RED_retrend`` adds it back). Both
round half away from zero and saturate as ``RED_round`` does. Scaling loses precision at write time; the
decoder reproduces what meflib returns for such a block, which is all any reader can recover.

The kernel is compiled with numba when it is installed, and releases the GIL, so channels can be decoded on
threads. Without numba a pure-NumPy version is used, which decodes all requested blocks in lockstep and is
much slower; numba has no wheels for Intel macOS, which is where that matters.
"""

from typing import NamedTuple

import numpy as np

try:
    import numba
except ImportError:  # pragma: no cover - exercised only where numba is not installed
    numba = None

NUMBA_AVAILABLE = numba is not None

# RED_POSITIVE_INFINITY and RED_NEGATIVE_INFINITY in meflib.h, the values RED_round saturates to.
_RED_POSITIVE_INFINITY = 0x7FFFFFFF
_RED_NEGATIVE_INFINITY = -0x7FFFFFFF
# RED_BLOCK_STATISTICS_BYTES symbols of at most 255 counts each.
_MAXIMUM_TOTAL_COUNTS = 256 * 255


class REDLayout(NamedTuple):
    """Where a format puts the parts of a RED block, in bytes from the start of the block."""

    header_bytes: int
    statistics_offset: int
    dummy_bytes: int
    keysample_bytes: int


# RED_BLOCK_HEADER_BYTES, RED_BLOCK_STATISTICS_OFFSET (meflib.h); no dummy byte; a keysample is an si4.
MEF3_RED_LAYOUT = REDLayout(header_bytes=304, statistics_offset=48, dummy_bytes=0, keysample_bytes=4)


# BLOCK_HEADER_BYTES and RED_STAT_MODEL_OFFSET (mef.h, mef_lib 2.1); RED_decompress_block skips one dummy byte
# after the header and reads a keysample as an si3.
MEF2_RED_LAYOUT = REDLayout(header_bytes=287, statistics_offset=31, dummy_bytes=1, keysample_bytes=3)


class REDDecodeError(ValueError):
    """A RED block whose coded data cannot be what its header describes."""


def _red_round(value):
    """``RED_round``: round half away from zero, saturating at the RED infinities."""
    if value >= 0.0:
        value += 0.5
        if value >= 2147483647.0:
            return _RED_POSITIVE_INFINITY
    else:
        value -= 0.5
        if value <= -2147483647.0:
            return _RED_NEGATIVE_INFINITY
    return int(value)


def _decode_blocks_python(
    buffer,
    block_starts,
    block_bytes,
    difference_counts,
    sample_counts,
    scale_factors,
    detrend_slopes,
    detrend_intercepts,
    header_bytes,
    statistics_offset,
    dummy_bytes,
    keysample_bytes,
    out,
):
    """
    Decode consecutive RED blocks of ``buffer`` into ``out``, returning -1, or the index of a block that is malformed.

    Written in the subset of Python numba compiles; it is also correct, if very slow, run as plain Python.
    Arithmetic is in int64 with explicit masks where meflib's ui4 arithmetic wraps, because numba would
    turn a mix of signed and unsigned integers into floats.
    """
    cumulative = np.zeros(257, np.int64)
    lookup = np.zeros(_MAXIMUM_TOTAL_COUNTS, np.int64)
    maximum_differences = 0
    for block in range(block_starts.shape[0]):
        if difference_counts[block] > maximum_differences:
            maximum_differences = difference_counts[block]
    differences = np.zeros(maximum_differences + 1, np.int64)
    keysample_sign = np.int64(1) << (8 * keysample_bytes - 1)
    keysample_span = np.int64(1) << (8 * keysample_bytes)

    position = 0
    for block in range(block_starts.shape[0]):
        number_of_samples = sample_counts[block]
        number_of_differences = difference_counts[block]
        if number_of_samples == 0:
            continue
        start = block_starts[block]
        end = start + block_bytes[block]
        statistics_start = start + statistics_offset

        # Cumulative symbol counts, and the symbol each cumulative count falls in: the symbol meflib finds
        # by searching the cumulative counts (the largest whose cumulative count does not exceed it).
        for symbol in range(256):
            cumulative[symbol + 1] = cumulative[symbol] + buffer[statistics_start + symbol]
        total = cumulative[256]
        if total == 0:
            return block
        for symbol in range(256):
            for count in range(cumulative[symbol], cumulative[symbol + 1]):
                lookup[count] = symbol

        # The range decoder, as in RED_decode. EXTRA_BITS is 7 and BOTTOM_VALUE is 0x800000.
        pointer = start + header_bytes + dummy_bytes
        if pointer >= end:
            return block
        in_byte = np.int64(buffer[pointer])
        pointer += 1
        low_bound = in_byte >> 1
        range_ = np.int64(1) << 7
        differences[0] = -128  # the first sample's keysample flag is implied, not coded
        for index in range(1, number_of_differences + 1):
            while range_ <= 0x800000:
                low_bound = ((low_bound << 8) | ((in_byte << 7) & 0xFF)) & 0xFFFFFFFF
                # Past the end of the block meflib feeds zero bytes rather than reading on.
                if pointer < end:
                    in_byte = np.int64(buffer[pointer])
                    pointer += 1
                else:
                    in_byte = 0
                low_bound |= in_byte >> 1
                range_ = range_ << 8
            range_per_count = range_ // total
            count = low_bound // range_per_count
            if count >= total:
                count = total - 1
            symbol = lookup[count]
            low_bound = (low_bound - range_per_count * cumulative[symbol]) & 0xFFFFFFFF
            if symbol < 255:
                range_ = range_per_count * buffer[statistics_start + symbol]
            else:
                range_ = range_ - range_per_count * cumulative[symbol]
            differences[index] = symbol - 256 if symbol > 127 else symbol

        # Differences to samples: a -128 introduces a little-endian keysample, anything else is added.
        cursor = 0
        current = np.int64(0)
        for sample in range(number_of_samples):
            if cursor > number_of_differences:
                return block
            difference = differences[cursor]
            if difference == -128:
                if cursor + keysample_bytes > number_of_differences:
                    return block
                value = np.int64(0)
                for byte in range(keysample_bytes):
                    value |= (differences[cursor + 1 + byte] & 0xFF) << (8 * byte)
                if value >= keysample_sign:
                    value -= keysample_span
                current = value
                cursor += 1 + keysample_bytes
            else:
                # si4 arithmetic, which wraps.
                current = ((current + difference + 0x80000000) & 0xFFFFFFFF) - 0x80000000
                cursor += 1
            out[position + sample] = current

        # RED_unscale, then RED_retrend, in that order, as RED_decode applies them.
        scale_factor = scale_factors[block]
        if scale_factor > 1.0:
            for sample in range(number_of_samples):
                out[position + sample] = _red_round(np.float64(out[position + sample]) * scale_factor)
        slope = detrend_slopes[block]
        intercept = detrend_intercepts[block]
        if slope != 0.0 or intercept != 0.0:
            abscissa = 0.0
            for sample in range(number_of_samples):
                abscissa += 1.0
                out[position + sample] = _red_round(np.float64(out[position + sample]) + slope * abscissa + intercept)

        position += number_of_samples
    return -1


if NUMBA_AVAILABLE:
    # Rebinding the global compiles the kernel's calls to _red_round too; numba resolves globals at compile time.
    _red_round = numba.njit(nogil=True, cache=True)(_red_round)
    _decode_blocks_compiled = numba.njit(nogil=True, cache=True)(_decode_blocks_python)


def _red_round_array(values: np.ndarray) -> np.ndarray:
    """``RED_round`` over an array of float64."""
    shifted = np.where(values >= 0.0, values + 0.5, values - 0.5)
    return np.trunc(np.clip(shifted, _RED_NEGATIVE_INFINITY, _RED_POSITIVE_INFINITY)).astype(np.int64)


def _decode_blocks_numpy(
    buffer,
    block_starts,
    block_bytes,
    difference_counts,
    sample_counts,
    scale_factors,
    detrend_slopes,
    detrend_intercepts,
    header_bytes,
    statistics_offset,
    dummy_bytes,
    keysample_bytes,
    out,
):
    """The same decode as :func:`_decode_blocks_python`, vectorised across blocks, which step through it in lockstep."""
    decoded = np.flatnonzero(sample_counts > 0)
    output_starts = np.concatenate([[0], np.cumsum(sample_counts)[:-1]])
    if decoded.size == 0:
        return -1
    starts = block_starts[decoded]
    ends = starts + block_bytes[decoded]
    differences_per_block = difference_counts[decoded]
    samples_per_block = sample_counts[decoded]
    number_of_blocks = decoded.size
    rows = np.arange(number_of_blocks)

    statistics = buffer[(starts + statistics_offset)[:, None] + np.arange(256)].astype(np.int64)
    cumulative = np.zeros((number_of_blocks, 257), np.int64)
    np.cumsum(statistics, axis=1, out=cumulative[:, 1:])
    total = cumulative[:, 256]
    if (total == 0).any():
        return int(decoded[np.flatnonzero(total == 0)[0]])
    lookup = np.empty((number_of_blocks, int(total.max())), np.int64)
    counts = np.arange(lookup.shape[1])
    for row in range(number_of_blocks):
        lookup[row] = np.searchsorted(cumulative[row], counts, side="right") - 1

    pointer = starts + header_bytes + dummy_bytes
    if (pointer >= ends).any():
        return int(decoded[np.flatnonzero(pointer >= ends)[0]])
    in_byte = buffer[pointer].astype(np.int64)
    pointer = pointer + 1
    low_bound = in_byte >> 1
    range_ = np.full(number_of_blocks, 1 << 7, np.int64)
    maximum_differences = int(differences_per_block.max())
    differences = np.zeros((number_of_blocks, maximum_differences + 1 + keysample_bytes), np.int64)
    differences[:, 0] = -128
    for index in range(1, maximum_differences + 1):
        active = differences_per_block >= index
        while True:
            renormalise = np.flatnonzero(active & (range_ <= 0x800000))
            if renormalise.size == 0:
                break
            shifted = ((low_bound[renormalise] << 8) | ((in_byte[renormalise] << 7) & 0xFF)) & 0xFFFFFFFF
            here = pointer[renormalise]
            inside = here < ends[renormalise]
            next_byte = np.where(inside, buffer[np.minimum(here, buffer.size - 1)], 0).astype(np.int64)
            pointer[renormalise] = here + inside
            in_byte[renormalise] = next_byte
            low_bound[renormalise] = shifted | (next_byte >> 1)
            range_[renormalise] = range_[renormalise] << 8
        current_rows = rows if active.all() else np.flatnonzero(active)
        range_per_count = range_[current_rows] // total[current_rows]
        count = np.minimum(low_bound[current_rows] // range_per_count, total[current_rows] - 1)
        symbol = lookup[current_rows, count]
        below = range_per_count * cumulative[current_rows, symbol]
        low_bound[current_rows] = (low_bound[current_rows] - below) & 0xFFFFFFFF
        range_[current_rows] = np.where(
            symbol < 255, range_per_count * statistics[current_rows, symbol], range_[current_rows] - below
        )
        differences[current_rows, index] = np.where(symbol > 127, symbol - 256, symbol)

    maximum_samples = int(samples_per_block.max())
    samples = np.zeros((number_of_blocks, maximum_samples), np.int64)
    cursor = np.zeros(number_of_blocks, np.int64)
    current = np.zeros(number_of_blocks, np.int64)
    byte_weights = np.int64(1) << (8 * np.arange(keysample_bytes, dtype=np.int64))
    keysample_sign = np.int64(1) << (8 * keysample_bytes - 1)
    keysample_span = np.int64(1) << (8 * keysample_bytes)
    malformed = np.zeros(number_of_blocks, bool)
    for sample in range(maximum_samples):
        active = samples_per_block > sample
        malformed |= active & (cursor > differences_per_block)
        difference = differences[rows, np.minimum(cursor, differences.shape[1] - 1)]
        keysample = active & (difference == -128)
        if keysample.any():
            malformed |= keysample & (cursor + keysample_bytes > differences_per_block)
            byte_columns = np.minimum(cursor[:, None] + 1 + np.arange(keysample_bytes), differences.shape[1] - 1)
            value = ((differences[rows[:, None], byte_columns] & 0xFF) * byte_weights).sum(axis=1)
            value = np.where(value >= keysample_sign, value - keysample_span, value)
            added = ((current + difference + 0x80000000) & 0xFFFFFFFF) - 0x80000000
            current = np.where(keysample, value, np.where(active, added, current))
            cursor = cursor + np.where(keysample, 1 + keysample_bytes, active.astype(np.int64))
        else:
            added = ((current + difference + 0x80000000) & 0xFFFFFFFF) - 0x80000000
            current = np.where(active, added, current)
            cursor = cursor + active
        samples[:, sample] = current
    if malformed.any():
        return int(decoded[np.flatnonzero(malformed)[0]])

    for row, block in enumerate(decoded):
        number_of_samples = int(samples_per_block[row])
        values = samples[row, :number_of_samples]
        if scale_factors[block] > 1.0:
            values = _red_round_array(values.astype(np.float64) * scale_factors[block])
        slope, intercept = detrend_slopes[block], detrend_intercepts[block]
        if slope != 0.0 or intercept != 0.0:
            abscissa = np.arange(1, number_of_samples + 1, dtype=np.float64)
            values = _red_round_array((values.astype(np.float64) + slope * abscissa) + intercept)
        out[output_starts[block] : output_starts[block] + number_of_samples] = values
    return -1


def decode_red_blocks(
    buffer: np.ndarray,
    *,
    block_starts: np.ndarray,
    block_bytes: np.ndarray,
    difference_counts: np.ndarray,
    sample_counts: np.ndarray,
    scale_factors: np.ndarray,
    detrend_slopes: np.ndarray,
    detrend_intercepts: np.ndarray,
    layout: REDLayout,
    use_numba: bool | None = None,
) -> np.ndarray:
    """
    Decode RED blocks held in ``buffer`` into one int32 array, the blocks' samples one after another.

    Parameters
    ----------
    buffer : numpy.ndarray of uint8
        Bytes holding the blocks, such as one contiguous read of a data file.
    block_starts, block_bytes : numpy.ndarray of int
        Each block's offset in ``buffer`` and its length, header included. Nothing is read past a block's end.
    difference_counts, sample_counts : numpy.ndarray of int
        Each block's number of coded differences and of samples, from its header.
    scale_factors, detrend_slopes, detrend_intercepts : numpy.ndarray of float
        Each block's scale factor, detrend slope and intercept, from its header. A scale factor above 1, or a
        nonzero slope or intercept, is undone as meflib does; 1, 0 and 0 leave the block as decoded.
    layout : REDLayout
        The format's block layout, such as :data:`MEF3_RED_LAYOUT`.
    use_numba : bool, optional
        Whether to use the compiled kernel. Defaults to using it when numba is installed.

    Returns
    -------
    numpy.ndarray of int32
        ``sample_counts.sum()`` samples.

    Raises
    ------
    REDDecodeError
        If a block is shorter than its header or lies outside ``buffer``, its coded data runs out before its
        samples do, or its symbol counts are all zero.
    """
    if use_numba is None:
        use_numba = NUMBA_AVAILABLE
    if use_numba and not NUMBA_AVAILABLE:
        raise ImportError("Decoding with use_numba=True requires numba. Install it with `pip install numba`.")

    sample_counts = np.ascontiguousarray(sample_counts, dtype=np.int64)
    arguments = (
        np.ascontiguousarray(buffer, dtype=np.uint8),
        np.ascontiguousarray(block_starts, dtype=np.int64),
        np.ascontiguousarray(block_bytes, dtype=np.int64),
        np.ascontiguousarray(difference_counts, dtype=np.int64),
        sample_counts,
        np.ascontiguousarray(scale_factors, dtype=np.float64),
        np.ascontiguousarray(detrend_slopes, dtype=np.float64),
        np.ascontiguousarray(detrend_intercepts, dtype=np.float64),
        int(layout.header_bytes),
        int(layout.statistics_offset),
        int(layout.dummy_bytes),
        int(layout.keysample_bytes),
    )
    out = np.empty(int(sample_counts.sum()), dtype=np.int32)
    if out.size == 0:
        return out

    # Checked here rather than in the kernels: the compiled one has no bounds checks, so a block shorter than
    # its header, or past the end of the buffer, would be read out of bounds, and the NumPy one would raise a
    # bare IndexError. The header holds the symbol statistics, and at least one byte of coded data follows it.
    buffer, block_starts, block_bytes = arguments[0], arguments[1], arguments[2]
    minimum_block_bytes = layout.header_bytes + layout.dummy_bytes + 1
    malformed = np.flatnonzero(
        (sample_counts > 0)
        & ((block_starts < 0) | (block_bytes < minimum_block_bytes) | (block_starts + block_bytes > buffer.size))
    )
    if malformed.size:
        block = int(malformed[0])
        raise REDDecodeError(
            f"RED block {block} of the {len(sample_counts)} requested cannot be decoded: it is stated to be "
            f"{block_bytes[block]} bytes at offset {block_starts[block]}, but a block holding samples is at least "
            f"{minimum_block_bytes} bytes and must lie within the {buffer.size} bytes read. The file is corrupt."
        )

    kernel = _decode_blocks_compiled if use_numba else _decode_blocks_numpy
    failed_block = kernel(*arguments, out)
    if failed_block >= 0:
        raise REDDecodeError(
            f"RED block {failed_block} of the {len(sample_counts)} requested cannot be decoded: its coded data "
            "ends before its samples do, or its symbol counts are all zero. The file is corrupt."
        )
    return out


# The Koopman CRC-32 both MEF versions checksum headers and blocks with: CRC_KOOPMAN32_KEY in mef.h is the
# table of the reflected polynomial Koopman32 (0xEB31D82E), which this rebuilds.
def _koopman_table() -> np.ndarray:
    table = np.zeros(256, dtype=np.int64)
    for index in range(256):
        crc = index
        for _ in range(8):
            crc = (crc >> 1) ^ (0xEB31D82E if crc & 1 else 0)
        table[index] = crc
    return table


_KOOPMAN_TABLE = _koopman_table()


def _crc_ranges_python(buffer, starts, ends, table, out):
    """``update_crc_32`` over each ``buffer[starts[i]:ends[i]]``, from 0xFFFFFFFF and without a final XOR."""
    for index in range(starts.shape[0]):
        crc = np.int64(0xFFFFFFFF)
        for position in range(starts[index], ends[index]):
            crc = (crc >> 8) ^ table[(crc ^ buffer[position]) & 0xFF]
        out[index] = crc


if NUMBA_AVAILABLE:
    _crc_ranges_compiled = numba.njit(nogil=True, cache=True)(_crc_ranges_python)


def _crc_ranges_numpy(buffer, starts, ends, table, out):
    """The same CRCs as :func:`_crc_ranges_python`, the ranges stepped through in lockstep."""
    crc = np.full(starts.shape[0], 0xFFFFFFFF, dtype=np.int64)
    lengths = ends - starts
    for step in range(int(lengths.max(initial=0))):
        active = np.flatnonzero(lengths > step)
        byte = buffer[starts[active] + step].astype(np.int64)
        crc[active] = (crc[active] >> 8) ^ table[(crc[active] ^ byte) & 0xFF]
    out[:] = crc


def koopman_crc32(
    buffer: np.ndarray, starts: np.ndarray, ends: np.ndarray, use_numba: bool | None = None
) -> np.ndarray:
    """
    The CRC meflib computes over each byte range ``buffer[starts[i]:ends[i]]``, as uint32.

    meflib's ``update_crc_32``: table-driven, starting from 0xFFFFFFFF, with no final XOR. The ranges must lie
    within ``buffer``.
    """
    if use_numba is None:
        use_numba = NUMBA_AVAILABLE
    if use_numba and not NUMBA_AVAILABLE:
        raise ImportError("Computing CRCs with use_numba=True requires numba. Install it with `pip install numba`.")
    starts = np.ascontiguousarray(starts, dtype=np.int64)
    ends = np.ascontiguousarray(ends, dtype=np.int64)
    if starts.size and (starts.min() < 0 or ends.max() > buffer.size or (ends < starts).any()):
        raise ValueError("A CRC range lies outside the buffer.")
    out = np.empty(starts.size, dtype=np.int64)
    kernel = _crc_ranges_compiled if use_numba else _crc_ranges_numpy
    kernel(np.ascontiguousarray(buffer, dtype=np.uint8), starts, ends, _KOOPMAN_TABLE, out)
    return out.astype(np.uint32)
