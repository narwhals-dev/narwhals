"""A minimal, pure-Python Parquet reader which decodes straight into a `dict[str, list]`.

Supported:

- flat and nested columns: lists (including the legacy two-level and unannotated
  layouts), structs, and maps (as lists of `{"key": ..., "value": ...}` structs);
- data pages V1 and V2, and dictionary pages;
- `PLAIN`, `PLAIN_DICTIONARY`/`RLE_DICTIONARY`, `RLE` (booleans), `BYTE_STREAM_SPLIT`,
  `DELTA_BINARY_PACKED`, `DELTA_LENGTH_BYTE_ARRAY` and `DELTA_BYTE_ARRAY` encodings;
- uncompressed, snappy, gzip, LZ4 (raw and Hadoop-framed) compression, and zstd with
  `compression.zstd` (Python 3.14+) or else the optional `backports.zstd`;
- strings, enums, JSON, decimals, dates, times, timestamps, (un)signed integers, and
  16-bit floats.

Anything else raises `NotImplementedError`. Corrupt data raises `ValueError`: page sizes,
page checksums (when the writer stored them) and row counts are all checked.

The format is specified at https://github.com/apache/parquet-format, and the metadata is
serialized with Thrift's compact protocol. Thrift structs are decoded generically, into
dicts keyed by field id, and only the fields this reader needs are looked up.
"""

from __future__ import annotations

import datetime as dt
import operator
import struct
import sys
import zlib
from contextlib import contextmanager
from decimal import Decimal
from functools import cache, partial
from itertools import accumulate, chain, islice, pairwise
from typing import TYPE_CHECKING, Any, NoReturn, TypeAlias

from pizzlies._plugin._vendored import check_column_names_are_unique

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence
    from typing import BinaryIO

    Converter = Callable[[Any], Any]
    Ranges: TypeAlias = "dict[int, list[tuple[int, int]]]"

MAGIC = b"PAR1"

# `Type`
BOOLEAN, INT32, INT64, INT96, FLOAT, DOUBLE, BYTE_ARRAY, FIXED_LEN_BYTE_ARRAY = range(8)
_STRUCT_FORMATS = {INT32: "i", INT64: "q", FLOAT: "f", DOUBLE: "d"}
_BITS = {INT32: 32, INT64: 64}

# `FieldRepetitionType`
REQUIRED, OPTIONAL, REPEATED = 0, 1, 2

# `PageType`
DATA_PAGE, DICTIONARY_PAGE, DATA_PAGE_V2 = 0, 2, 3

# `Encoding`
PLAIN, PLAIN_DICTIONARY, RLE, RLE_DICTIONARY = 0, 2, 3, 8
DELTA_BINARY_PACKED, DELTA_LENGTH_BYTE_ARRAY, DELTA_BYTE_ARRAY = 5, 6, 7
BYTE_STREAM_SPLIT = 9
_ENCODING_NAMES = {4: "BIT_PACKED"}

# `CompressionCodec`
UNCOMPRESSED, SNAPPY, GZIP, LZ4, ZSTD, LZ4_RAW = 0, 1, 2, 5, 6, 7
_CODEC_NAMES = {3: "LZO", 4: "BROTLI"}

# `ConvertedType`, for files which predate logical types
_UTF8, _MAP, _MAP_KEY_VALUE, _LIST, _ENUM, _DECIMAL = 0, 1, 2, 3, 4, 5
_DATE, _TIME_MILLIS, _TIME_MICROS, _TIMESTAMP_MILLIS, _TIMESTAMP_MICROS = 6, 7, 8, 9, 10
_JSON = 19
_UNSIGNED_BITS = {11: 8, 12: 16, 13: 32, 14: 64}

# `LogicalType` union members
_L_STRING, _L_MAP, _L_LIST, _L_ENUM, _L_DECIMAL, _L_DATE = 1, 2, 3, 4, 5, 6
_L_TIME, _L_TIMESTAMP, _L_INTEGER, _L_JSON, _L_FLOAT16 = 7, 8, 10, 12, 15
_L_UNSUPPORTED = {16: "VARIANT"}

# Annotations which are only valid on INT32 and INT64 columns.
_INT_ONLY = {_DATE, _TIME_MILLIS, _TIME_MICROS, _TIMESTAMP_MILLIS, _TIMESTAMP_MICROS}
_INT_ONLY |= _UNSIGNED_BITS.keys()
_L_INT_ONLY = {_L_DATE, _L_TIME, _L_TIMESTAMP, _L_INTEGER}

# `TimeUnit` union members, to a divisor which turns the stored value into microseconds
_MILLIS, _MICROS, _NANOS = 1, 2, 3

_EPOCH_ORDINAL = dt.date(1970, 1, 1).toordinal()
_JULIAN_EPOCH = 2_440_588  # Julian day number of 1970-01-01, for INT96 timestamps

_CORRUPT_MESSAGE = "Truncated or invalid Parquet data, the file may be corrupt."

# What malformed metadata or data surfaces as, deep in the decoder: e.g. a missing
# Thrift field (`KeyError`), one of the wrong type (`TypeError`), or reading past the
# end of a buffer (`IndexError`).
_CORRUPTION_ERRORS = (
    AttributeError,
    IndexError,
    KeyError,
    RecursionError,
    TypeError,
    UnicodeDecodeError,  # e.g. in a column name: invalid values raise a clearer error
    ZeroDivisionError,
    struct.error,
)


class _Reader:
    """Cursor over a buffer, for Thrift's compact protocol."""

    __slots__ = ("buf", "pos")

    def __init__(self, buf: bytes, pos: int = 0) -> None:
        self.buf = buf
        self.pos = pos

    def byte(self) -> int:
        value = self.buf[self.pos]
        self.pos += 1
        return value

    def varint(self) -> int:
        result = shift = 0
        while True:
            byte = self.buf[self.pos]
            self.pos += 1
            result |= (byte & 0x7F) << shift
            if not byte & 0x80:
                return result
            shift += 7

    def zigzag(self) -> int:
        n = self.varint()
        return (n >> 1) ^ -(n & 1)

    def read(self, n: int) -> bytes:
        if n < 0 or self.pos + n > len(self.buf):
            _raise_corrupt()
        value = self.buf[self.pos : self.pos + n]
        self.pos += n
        return value


def _read_value(r: _Reader, kind: int) -> Any:  # noqa: C901, PLR0911
    if kind in {1, 2}:  # BOOLEAN_TRUE, BOOLEAN_FALSE (as a struct field)
        return kind == 1
    if kind == 3:  # BYTE
        return r.byte()
    if kind in {4, 5, 6}:  # I16, I32, I64
        return r.zigzag()
    if kind == 7:  # DOUBLE
        return struct.unpack("<d", r.read(8))[0]
    if kind == 8:  # BINARY
        return r.read(r.varint())
    if kind in {9, 10}:  # LIST, SET
        header = r.byte()
        size, element_kind = header >> 4, header & 0x0F
        if size == 15:
            size = r.varint()
        if element_kind in {1, 2}:  # booleans in a list are one byte each
            return [r.byte() == 1 for _ in range(size)]
        return [_read_value(r, element_kind) for _ in range(size)]
    if kind == 11:  # MAP
        size = r.varint()
        if not size:
            return {}
        kinds = r.byte()
        return {
            _read_value(r, kinds >> 4): _read_value(r, kinds & 0x0F) for _ in range(size)
        }
    if kind == 12:  # STRUCT
        return _read_struct(r)
    raise ValueError(_CORRUPT_MESSAGE)


def _read_struct(r: _Reader) -> dict[int, Any]:
    fields: dict[int, Any] = {}
    field_id = 0
    while header := r.byte():
        delta = header >> 4
        field_id = field_id + delta if delta else r.zigzag()
        fields[field_id] = _read_value(r, header & 0x0F)
    return fields


def _copy_back(out: bytearray, offset: int, length: int) -> None:
    """Append `length` bytes copied from `offset` bytes back, as LZ77 compressors do."""
    if not 0 < offset <= len(out):
        _raise_corrupt()
    start = len(out) - offset
    if offset >= length:
        out += out[start : start + length]
    else:  # the copy overlaps what it produces, so it repeats the last `offset` bytes
        out += (out[start:] * (length // offset + 1))[:length]


def _snappy_decompress(data: bytes, size: int) -> bytes:
    r = _Reader(data)
    if r.varint() != size:
        _raise_corrupt()
    pos, end, out = r.pos, len(data), bytearray()
    while pos < end:  # no bomb: a tag of 1+ bytes yields at most 64 bytes
        tag = data[pos]
        pos += 1
        kind = tag & 0b11
        if kind == 0:  # literal
            length = tag >> 2
            if length >= 60:
                n_bytes = length - 59
                length = int.from_bytes(data[pos : pos + n_bytes], "little")
                pos += n_bytes
            length += 1
            out += data[pos : pos + length]
            pos += length
            continue
        if kind == 1:  # copy, 1-byte offset
            length = ((tag >> 2) & 0b111) + 4
            offset = ((tag >> 5) << 8) | data[pos]
            pos += 1
        else:  # copy, 2- or 4-byte offset
            n_bytes = 2 if kind == 2 else 4
            length = (tag >> 2) + 1
            offset = int.from_bytes(data[pos : pos + n_bytes], "little")
            pos += n_bytes
        # `_copy_back`, inlined as this loop runs once per tag.
        if (start := len(out) - offset) < 0 or not offset:
            _raise_corrupt()
        if offset >= length:
            out += out[start : start + length]
        else:  # the copy overlaps what it produces, so it repeats the last `offset` bytes
            out += (out[start:] * (length // offset + 1))[:length]
    if pos != end:  # a literal or an offset ran past the end
        _raise_corrupt()
    return bytes(out)


def _lz4_length(data: bytes, pos: int, length: int) -> tuple[int, int]:
    """Read the extra bytes of an LZ4 length whose 4-bit nibble is saturated."""
    if length == 15:
        while True:
            byte = data[pos]
            pos += 1
            length += byte
            if byte != 255:
                break
    return length, pos


def _lz4_block_decompress(data: bytes, size: int) -> bytes:
    pos, end, out = 0, len(data), bytearray()
    while pos < end and len(out) <= size:
        token = data[pos]
        length, pos = _lz4_length(data, pos + 1, token >> 4)
        if pos + length > end:
            _raise_corrupt()
        out += data[pos : pos + length]
        pos += length
        if pos == end:  # the last sequence has literals only
            break
        if pos + 2 > end:
            _raise_corrupt()
        offset = int.from_bytes(data[pos : pos + 2], "little")
        length, pos = _lz4_length(data, pos + 2, token & 0x0F)
        _copy_back(out, offset, length + 4)
    return bytes(out)


def _hadoop_lz4_decompress(data: bytes, size: int) -> bytes | None:
    """Hadoop's framing of LZ4 blocks, each prefixed by its sizes, or `None` if not it."""
    pos, out = 0, bytearray()
    while pos < len(data):
        if pos + 8 > len(data):
            return None
        expected = int.from_bytes(data[pos : pos + 4], "big")
        compressed = int.from_bytes(data[pos + 4 : pos + 8], "big")
        pos += 8
        if pos + compressed > len(data) or len(out) + expected > size:
            return None
        try:
            block = _lz4_block_decompress(data[pos : pos + compressed], expected)
        except (ValueError, IndexError):
            return None
        if len(block) != expected:
            return None
        out += block
        pos += compressed
    return bytes(out)


def _stream_decompress(
    decompressor: Callable[[], Any], error: type[Exception], data: bytes, size: int
) -> bytes:
    """Decompress every concatenated member (gzip) or frame (zstd), up to `size` bytes."""
    out = bytearray()
    while data:
        d = decompressor()
        try:
            # At most one byte more than expected, enough to detect a size mismatch
            # without inflating a decompression bomb.
            out += d.decompress(data, size + 1 - len(out))
        except error:
            _raise_corrupt()
        if not d.eof or len(out) > size:
            _raise_corrupt()
        data = d.unused_data
    return bytes(out)


def _zstd() -> Any:
    if sys.version_info >= (3, 14):
        from compression import zstd
    else:
        try:  # optional: `pip install pizzlies[zstd]`
            from backports import zstd  # type: ignore[import-not-found,unused-ignore]  # pyright: ignore[reportMissingImports]
        except ImportError:
            msg = (
                "Reading zstd-compressed Parquet needs Python 3.14+, or the "
                "`backports.zstd` package (`pip install 'pizzlies[zstd]'`)."
            )
            raise NotImplementedError(msg) from None
    return zstd


def _decompress(codec: int, data: bytes, size: int) -> bytes:
    """Decompress `data`, which must decompress to `size` bytes."""
    if codec == UNCOMPRESSED:
        return data
    if codec == SNAPPY:
        out = _snappy_decompress(data, size)
    elif codec == GZIP:

        def gzip() -> Any:
            return zlib.decompressobj(wbits=zlib.MAX_WBITS | 32)

        out = _stream_decompress(gzip, zlib.error, data, size)
    elif codec == ZSTD:
        zstd = _zstd()
        out = _stream_decompress(zstd.ZstdDecompressor, zstd.ZstdError, data, size)
    elif codec == LZ4_RAW:
        out = _lz4_block_decompress(data, size)
    elif codec == LZ4:
        # Ambiguous: Hadoop's framing, or else (as old parquet-cpp wrote) a raw block.
        hadoop = _hadoop_lz4_decompress(data, size)
        out = _lz4_block_decompress(data, size) if hadoop is None else hadoop
    else:
        msg = f"Parquet compression codec {_CODEC_NAMES.get(codec, codec)} is not supported."
        raise NotImplementedError(msg)
    if len(out) != size:
        _raise_corrupt()
    return out


@cache
def _bit_lanes(bit_width: int) -> list[tuple[int, list[list[int]]]]:
    """How to read each of the 8 values of a bit-packed group, see `_unpack_bits`.

    For value `j`: the offset of its first byte in the group, and one 256-entry table
    per byte it spans, mapping that byte to its contribution to the value.
    """
    mask = (1 << bit_width) - 1
    lanes = []
    for j in range(8):
        first, shift = divmod(j * bit_width, 8)
        n_bytes = (shift + bit_width + 7) // 8
        tables = [
            [((byte << (8 * k)) >> shift) & mask for byte in range(256)]
            for k in range(n_bytes)
        ]
        lanes.append((first, tables))
    return lanes


def _unpack_bits(data: bytes, bit_width: int) -> list[int]:
    """Unpack groups of 8 little-endian `bit_width`-bit values, `bit_width` bytes each.

    Value `j` of every group sits at the same byte offset and bit shift, so all of
    them are read at once from strided slices, through lookup tables: there's no
    Python-level work per value.
    """
    lanes = []
    for first, (table, *more_tables) in _bit_lanes(bit_width):
        lane: Iterator[int] = map(table.__getitem__, data[first::bit_width])
        for k, extra in enumerate(more_tables, start=1):
            lane = map(
                operator.add, lane, map(extra.__getitem__, data[first + k :: bit_width])
            )
        lanes.append(lane)
    return list(chain.from_iterable(zip(*lanes, strict=True)))


def _decode_rle_bitpacked(
    buf: bytes, pos: int, end: int, bit_width: int, count: int
) -> list[int]:
    """Decode `count` values of the RLE / bit-packing hybrid encoding."""
    if bit_width == 0:
        return [0] * count
    byte_width = (bit_width + 7) // 8
    # Writers interleave many short bit-packed and RLE runs. Unpacking every
    # bit-packed run in one go is much cheaper than unpacking each separately:
    # `runs` holds `(value, n)` for RLE runs, and `(None, n)` for bit-packed ones.
    runs: list[tuple[int | None, int]] = []
    packed: list[bytes] = []
    n_values = 0
    end = min(end, len(buf))
    while n_values < count and pos < end:
        r = _Reader(buf, pos)
        header = r.varint()
        pos = r.pos
        if header & 1:  # bit-packed groups of 8 values, `bit_width` bytes per group
            n_bytes = (header >> 1) * bit_width
            if pos + n_bytes > end:
                _raise_corrupt()
            packed.append(buf[pos : pos + n_bytes])
            pos += n_bytes
            runs.append((None, (header >> 1) * 8))
        else:  # a run of one repeated value
            if (n := header >> 1) > count - n_values or pos + byte_width > end:
                _raise_corrupt()
            runs.append((int.from_bytes(buf[pos : pos + byte_width], "little"), n))
            pos += byte_width
        n_values += runs[-1][1]
    unpacked = _unpack_bits(b"".join(packed), bit_width) if packed else []
    if len(runs) == 1 and runs[0][0] is None:
        out = unpacked
    else:
        out = []
        offset = 0
        for value, n in runs:
            if value is None:
                out += unpacked[offset : offset + n]
                offset += n
            else:
                out += [value] * n
    if len(out) < count:
        _raise_corrupt()
    del out[count:]
    return out


def _decode_delta_binary_packed(
    buf: bytes, pos: int, bits: int, count: int
) -> tuple[list[int], int]:
    """Decode `count` `bits`-bit integers, and return them with the position after them."""
    r = _Reader(buf, pos)
    block_size, n_miniblocks, total, first = (
        r.varint(),
        r.varint(),
        r.varint(),
        r.zigzag(),
    )
    # Checking the count first bounds the work: width-0 miniblocks take no space.
    if (
        total != count
        or not n_miniblocks
        or block_size % n_miniblocks
        or block_size // n_miniblocks % 8
    ):
        _raise_corrupt()
    per_miniblock = block_size // n_miniblocks
    deltas: list[int] = []
    remaining = total - 1
    while remaining > 0:
        min_delta = r.zigzag()
        for width in r.read(n_miniblocks):
            if remaining <= 0:  # the last block's unneeded miniblocks hold no data
                break
            if width > bits:
                _raise_corrupt()
            packed = r.read(per_miniblock * width // 8)
            n = min(per_miniblock, remaining)
            unpacked = _unpack_bits(packed, width)[:n] if width else [0] * n
            deltas += [min_delta + delta for delta in unpacked]
            remaining -= n
    values = list(accumulate(deltas, initial=first)) if total else []
    # Deltas may overflow, so the running sums are wrapped to `bits` bits, as stored.
    half = 1 << (bits - 1)
    if any(not -half <= value < half for value in values):
        mask = (1 << bits) - 1
        values = [((value + half) & mask) - half for value in values]
    return values, r.pos


def _decode_delta_length_byte_arrays(
    buf: bytes, pos: int, count: int
) -> tuple[list[bytes], int]:
    lengths, pos = _decode_delta_binary_packed(buf, pos, 32, count)
    if any(length < 0 for length in lengths):
        _raise_corrupt()
    ends = list(accumulate(lengths, initial=pos))
    _check_available(buf, ends[-1])
    return [buf[start:end] for start, end in pairwise(ends)], ends[-1]


def _decode_delta_byte_arrays(buf: bytes, pos: int, count: int) -> list[bytes]:
    prefixes, pos = _decode_delta_binary_packed(buf, pos, 32, count)
    suffixes, _ = _decode_delta_length_byte_arrays(buf, pos, count)
    out = []
    previous = b""
    for prefix, suffix in zip(prefixes, suffixes, strict=True):
        if not 0 <= prefix <= len(previous):
            _raise_corrupt()
        previous = previous[:prefix] + suffix
        out.append(previous)
    return out


def _byte_stream_split(buf: bytes, pos: int, count: int, width: int) -> bytes:
    """Interleave the `width` streams back: stream `k` holds byte `k` of every value."""
    _check_available(buf, pos + count * width)
    out = bytearray(count * width)
    for k in range(width):
        out[k::width] = buf[pos + k * count : pos + (k + 1) * count]
    return bytes(out)


def _raise_corrupt() -> NoReturn:
    raise ValueError(_CORRUPT_MESSAGE)


def _check_available(buf: bytes, end: int) -> None:
    if end > len(buf):
        _raise_corrupt()


_read_uint32 = struct.Struct("<I").unpack_from


def _decode_byte_arrays(buf: bytes, pos: int, count: int, *, utf8: bool) -> list[Any]:
    """PLAIN `BYTE_ARRAY` values: each is a 4-byte length followed by its bytes.

    With `utf8`, values are decoded to `str`. If the data is pure ASCII, bytes and
    characters line up, so it's decoded once and sliced, rather than value by value.
    """
    data = buf[pos:]
    source: bytes | str = data.decode("ascii") if utf8 and data.isascii() else data
    out: list[Any] = []
    append = out.append
    pos = 0
    try:
        for _ in range(count):
            (length,) = _read_uint32(data, pos)
            pos += 4
            append(source[pos : pos + length])
            pos += length
    except struct.error:
        _raise_corrupt()
    _check_available(data, pos)
    if utf8 and source is data:
        return list(map(bytes.decode, out))
    return out


def _convert(values: list[Any], column: _Column) -> list[Any]:
    if column.convert is None:
        return values
    try:
        return list(map(column.convert, values))
    except UnicodeDecodeError as exc:
        msg = f"Parquet column {column.name!r} holds invalid UTF-8: {exc}"
        raise ValueError(msg) from exc
    except (OverflowError, ValueError) as exc:  # e.g. a timestamp beyond the year 9999
        msg = (
            f"Parquet column {column.name!r} holds a value Python can't represent: {exc}"
        )
        raise ValueError(msg) from exc


def _decode_plain_values(page: bytes, pos: int, count: int, column: _Column) -> list[Any]:
    """`_decode_plain`, converted to Python values."""
    if column.physical == BYTE_ARRAY and column.convert is bytes.decode:
        try:
            return _decode_byte_arrays(page, pos, count, utf8=True)
        except UnicodeDecodeError as exc:
            msg = f"Parquet column {column.name!r} holds invalid UTF-8: {exc}"
            raise ValueError(msg) from exc
    values = _decode_plain(page, pos, column.physical, count, column.type_length)
    return _convert(values, column)


def _decode_plain(
    buf: bytes, pos: int, physical: int, count: int, type_length: int | None
) -> list[Any]:
    if count < 0:
        _raise_corrupt()
    if physical == BYTE_ARRAY:
        return _decode_byte_arrays(buf, pos, count, utf8=False)
    if fmt := _STRUCT_FORMATS.get(physical):
        fmt = f"<{count}{fmt}"
        _check_available(buf, pos + struct.calcsize(fmt))
        return list(struct.unpack_from(fmt, buf, pos))
    if physical == BOOLEAN:
        _check_available(buf, pos + (count + 7) // 8)
        return [bool((buf[pos + i // 8] >> (i % 8)) & 1) for i in range(count)]
    width = _value_width(physical, type_length)
    _check_available(buf, pos + count * width)
    return [buf[pos + i * width : pos + (i + 1) * width] for i in range(count)]


def _value_width(physical: int, type_length: int | None) -> int:
    """The size of a fixed-width value, in bytes."""
    if physical in _STRUCT_FORMATS:
        return struct.calcsize(_STRUCT_FORMATS[physical])
    if physical == INT96:
        return 12
    if physical != FIXED_LEN_BYTE_ARRAY or type_length is None or type_length < 0:
        _raise_corrupt()
    return type_length


def _timestamp_converter(unit: int, *, utc: bool) -> Converter:
    epoch = dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc if utc else None)
    delta = dt.timedelta
    if unit == _MILLIS:
        return lambda v: epoch + delta(milliseconds=v)
    if unit == _MICROS:
        return lambda v: epoch + delta(microseconds=v)
    return lambda v: epoch + delta(microseconds=v // 1000)


def _time_converter(unit: int) -> Converter:
    divisor = {_MILLIS: 1, _MICROS: 1, _NANOS: 1000}[unit]
    scale = 1000 if unit == _MILLIS else 1

    def convert(value: int) -> dt.time:
        seconds, micros = divmod(value * scale // divisor, 1_000_000)
        minutes, seconds = divmod(seconds, 60)
        return dt.time(*divmod(minutes, 60), seconds, micros)

    return convert


def _int96_to_datetime(raw: bytes) -> dt.datetime:
    nanos, julian_day = struct.unpack("<qi", raw)
    return dt.datetime(1970, 1, 1) + dt.timedelta(
        days=julian_day - _JULIAN_EPOCH, microseconds=nanos // 1000
    )


def _float16(raw: bytes) -> float:
    value: float = struct.unpack("<e", raw)[0]
    return value


def _converter(element: dict[int, Any]) -> Converter | None:  # noqa: C901, PLR0911, PLR0912
    """Map a stored (physical) value to a Python value, or `None` for no conversion."""
    physical = element[1]
    logical: dict[int, Any] = element.get(10) or {}
    converted: int = element.get(6, -1)
    if unsupported := [_L_UNSUPPORTED[k] for k in logical if k in _L_UNSUPPORTED]:
        msg = f"Parquet logical type {unsupported[0]} is not supported."
        raise NotImplementedError(msg)
    if {_L_STRING, _L_ENUM, _L_JSON} & logical.keys() or converted in {
        _UTF8,
        _ENUM,
        _JSON,
    }:
        return bytes.decode
    if _L_FLOAT16 in logical:
        if physical != FIXED_LEN_BYTE_ARRAY or element.get(2) != 2:
            _raise_corrupt()
        return _float16
    if _L_DECIMAL in logical or converted == _DECIMAL:
        scale = (
            logical[_L_DECIMAL].get(1, 0) if _L_DECIMAL in logical else element.get(7, 0)
        )
        # Built from a string, as `scaleb` would round to the context's precision.
        if physical in {BYTE_ARRAY, FIXED_LEN_BYTE_ARRAY}:
            return lambda b: Decimal(f"{int.from_bytes(b, 'big', signed=True)}E-{scale}")
        return lambda v: Decimal(f"{v}E-{scale}")
    if physical not in {INT32, INT64} and (
        _L_INT_ONLY & logical.keys() or converted in _INT_ONLY
    ):
        msg = "Parquet date, time, timestamp and integer types must be stored as INT32 or INT64."
        raise NotImplementedError(msg)
    if _L_DATE in logical or converted == _DATE:
        return lambda d: dt.date.fromordinal(_EPOCH_ORDINAL + d)
    if _L_TIMESTAMP in logical:
        timestamp = logical[_L_TIMESTAMP]
        (unit,) = timestamp[2]
        return _timestamp_converter(unit, utc=timestamp.get(1, False))
    if converted in {_TIMESTAMP_MILLIS, _TIMESTAMP_MICROS}:
        unit = _MILLIS if converted == _TIMESTAMP_MILLIS else _MICROS
        return _timestamp_converter(unit, utc=True)
    if _L_TIME in logical:
        (unit,) = logical[_L_TIME][2]
        return _time_converter(unit)
    if converted in {_TIME_MILLIS, _TIME_MICROS}:
        return _time_converter(_MILLIS if converted == _TIME_MILLIS else _MICROS)
    if _L_INTEGER in logical:
        bits, signed = logical[_L_INTEGER].get(1, 64), logical[_L_INTEGER].get(2, True)
    else:
        bits, signed = _UNSIGNED_BITS.get(converted, 64), converted not in _UNSIGNED_BITS
    if not signed:
        mask = (1 << bits) - 1
        return lambda v: v & mask
    if physical == INT96:
        return _int96_to_datetime
    return None


class _Column:
    """A leaf of the schema: a physical column, with one column chunk per row group."""

    __slots__ = ("convert", "max_def", "max_rep", "name", "physical", "type_length")

    def __init__(
        self,
        element: dict[int, Any],
        *,
        name: str | None = None,
        max_def: int | None = None,
        max_rep: int = 0,
    ) -> None:
        self.name: str = name or element[4].decode()
        self.physical: int = element[1]
        self.type_length: int | None = element.get(2)
        self.max_def = int(element.get(3) == OPTIONAL) if max_def is None else max_def
        self.max_rep = max_rep
        self.convert = _converter(element)


# How a top-level column's values are assembled from its leaf columns' levels and values.
# `leaves` are the indices of the leaf columns below each node. `present` is the
# definition level from which an optional node isn't null, `None` if it's required.
class _LeafShape:
    __slots__ = ("index", "leaves")

    def __init__(self, index: int) -> None:
        self.index = index
        self.leaves: list[int] = [index]


class _StructShape:
    __slots__ = ("fields", "leaves", "present")

    def __init__(self, fields: list[tuple[str, Shape]], present: int | None) -> None:
        if not fields:
            _raise_corrupt()
        self.fields = fields
        self.present = present
        self.leaves: list[int] = [i for _, shape in fields for i in shape.leaves]


class _ListShape:
    """A list, whose `element`s are at repetition level `rep`.

    A list is empty below definition level `nonempty`, that of its repeated node.
    """

    __slots__ = ("element", "leaves", "nonempty", "present", "rep")

    def __init__(
        self, element: Shape, *, present: int | None, nonempty: int, rep: int
    ) -> None:
        self.element = element
        self.present = present
        self.nonempty = nonempty
        self.rep = rep
        self.leaves: list[int] = element.leaves


Shape: TypeAlias = _LeafShape | _StructShape | _ListShape
_SchemaNode: TypeAlias = "tuple[dict[int, Any], list[_SchemaNode]]"


class _Schema:
    """The top-level fields of a file's schema, and its leaf columns."""

    def __init__(self, elements: list[dict[int, Any]]) -> None:
        root, end = self._tree(elements, 0)
        if end != len(elements):
            _raise_corrupt()
        self.leaves: list[_Column] = []
        self.fields: list[tuple[str, Shape]] = [
            (element[4].decode(), self._node(element, children, 0, 0, ()))
            for element, children in root[1]
        ]

    @classmethod
    def _tree(cls, elements: list[dict[int, Any]], i: int) -> tuple[_SchemaNode, int]:
        """Rebuild the tree from its depth-first list, where groups count their children."""
        element = elements[i]
        children: list[_SchemaNode] = []
        i += 1
        for _ in range(element.get(5, 0)):
            child, i = cls._tree(elements, i)
            children.append(child)
        return (element, children), i

    def _node(
        self,
        element: dict[int, Any],
        children: list[_SchemaNode],
        parent_def: int,
        parent_rep: int,
        path: tuple[str, ...],
    ) -> Shape:
        repetition = element.get(3, REQUIRED)
        level_def = parent_def + (repetition != REQUIRED)
        level_rep = parent_rep + (repetition == REPEATED)
        path = (*path, element[4].decode())
        if repetition == REPEATED:
            # Not part of a `LIST` or `MAP`: a required list of required elements.
            element_shape = self._inner(
                element, children, level_def, level_rep, path, present=None
            )
            return _ListShape(
                element_shape, present=None, nonempty=level_def, rep=level_rep
            )
        present = level_def if repetition == OPTIONAL else None
        return self._inner(element, children, level_def, level_rep, path, present=present)

    def _inner(
        self,
        element: dict[int, Any],
        children: list[_SchemaNode],
        level_def: int,
        level_rep: int,
        path: tuple[str, ...],
        *,
        present: int | None,
    ) -> Shape:
        """The shape of a node whose own repetition was accounted for by the caller."""
        if 5 not in element:
            self.leaves.append(
                _Column(
                    element, name=".".join(path), max_def=level_def, max_rep=level_rep
                )
            )
            return _LeafShape(len(self.leaves) - 1)
        logical: dict[int, Any] = element.get(10) or {}
        if unsupported := [_L_UNSUPPORTED[k] for k in logical if k in _L_UNSUPPORTED]:
            msg = f"Parquet logical type {unsupported[0]} is not supported."
            raise NotImplementedError(msg)
        converted = element.get(6)
        repeated = len(children) == 1 and children[0][0].get(3) == REPEATED
        if _L_LIST in logical or converted == _LIST:
            if not repeated:
                msg = f"Unsupported layout of the Parquet LIST {'.'.join(path)!r}."
                raise NotImplementedError(msg)
            ((child, grandchildren),) = children
            name = child[4].decode()
            child_def, child_rep = level_def + 1, level_rep + 1
            child_path = (*path, name)
            if (
                5 not in child
                or len(grandchildren) != 1
                or grandchildren[0][0].get(3) == REPEATED
                or name in {"array", f"{path[-1]}_tuple"}
            ):
                # Legacy two-level list: the repeated node is the element itself.
                element_shape = self._inner(
                    child, grandchildren, child_def, child_rep, child_path, present=None
                )
            else:
                ((leaf, leaf_children),) = grandchildren
                element_shape = self._node(
                    leaf, leaf_children, child_def, child_rep, child_path
                )
            return _ListShape(
                element_shape, present=present, nonempty=child_def, rep=child_rep
            )
        if (_L_MAP in logical or converted in {_MAP, _MAP_KEY_VALUE}) and repeated:
            ((child, grandchildren),) = children
            if not 1 <= len(grandchildren) <= 2:
                msg = f"Unsupported layout of the Parquet MAP {'.'.join(path)!r}."
                raise NotImplementedError(msg)
            child_def, child_rep = level_def + 1, level_rep + 1
            child_path = (*path, child[4].decode())
            entry = [
                (name, self._node(*node, child_def, child_rep, child_path))
                for name, node in zip(("key", "value"), grandchildren, strict=False)
            ]
            # Without values, as PyArrow does, a list of the keys.
            element_shape = _StructShape(entry, None) if len(entry) == 2 else entry[0][1]
            return _ListShape(
                element_shape, present=present, nonempty=child_def, rep=child_rep
            )
        fields = [
            (
                child[4].decode(),
                self._node(child, grandchildren, level_def, level_rep, path),
            )
            for child, grandchildren in children
        ]
        return _StructShape(fields, present)


class _LeafData:
    """The levels and (non-null) values of a leaf column, in one row group."""

    __slots__ = ("defs", "max_def", "reps", "values")

    def __init__(
        self,
        defs: list[int] | None,
        reps: list[int] | None,
        values: list[Any],
        max_def: int,
    ) -> None:
        self.defs = defs
        self.reps = reps
        self.values = values
        self.max_def = max_def

    def __len__(self) -> int:
        return len(self.values) if self.defs is None else len(self.defs)

    def aligned(self) -> list[Any]:
        """One value per level, `None` where it's null (or an ancestor is)."""
        if self.defs is None:
            return self.values
        it = iter(self.values)
        max_def = self.max_def
        return [next(it) if level == max_def else None for level in self.defs]


def _levels(levels: list[int] | None) -> list[int]:
    if levels is None:  # e.g. a nested schema whose leaves have no levels
        _raise_corrupt()
    return levels


def _row_ranges(leaf: _LeafData, n_rows: int) -> list[tuple[int, int]]:
    """The `[start, end)` level ranges of each row: a new row starts at repetition 0."""
    if leaf.reps is None:
        if len(leaf) != n_rows:
            _raise_corrupt()
        return [(i, i + 1) for i in range(n_rows)]
    starts = [i for i, level in enumerate(leaf.reps) if level == 0]
    if len(starts) != n_rows or (starts and starts[0] != 0):
        _raise_corrupt()
    return list(pairwise([*starts, len(leaf)]))


def _split_lists(
    leaf: _LeafData, ranges: list[tuple[int, int]], shape: _ListShape
) -> tuple[list[tuple[int, int]], list[int | None]]:
    """Split each list's level range into its elements' ranges.

    Also return each list's length, or `None` for a null list.
    """
    defs, reps = _levels(leaf.defs), _levels(leaf.reps)
    present, nonempty, rep = shape.present, shape.nonempty, shape.rep
    elements: list[tuple[int, int]] = []
    lengths: list[int | None] = []
    for start, end in ranges:
        level = defs[start]
        if level < nonempty:
            lengths.append(None if present is not None and level < present else 0)
            continue
        n_before = len(elements)
        element_start = start
        for i in range(start + 1, end):
            if reps[i] <= rep:  # a new element of this list (not of a nested one)
                elements.append((element_start, i))
                element_start = i
        elements.append((element_start, end))
        lengths.append(len(elements) - n_before)
    return elements, lengths


def _struct_presence(
    leaves: list[int], present: int, ranges: Ranges, data: dict[int, _LeafData]
) -> list[bool]:
    """Whether each of an optional struct's rows isn't null."""
    is_present: list[bool] | None = None
    for index in leaves:
        defs = _levels(data[index].defs)
        leaf_present = [defs[start] >= present for start, _ in ranges[index]]
        if is_present is None:
            is_present = leaf_present
        elif leaf_present != is_present:  # it's null in all its leaves, or in none
            _raise_corrupt()
    return is_present or []


def _assemble(shape: Shape, ranges: Ranges, data: dict[int, _LeafData]) -> list[Any]:
    """Build one value per range, from the levels and values of `shape`'s leaves."""
    if isinstance(shape, _LeafShape):
        leaf_ranges = ranges[shape.index]
        if any(end - start != 1 for start, end in leaf_ranges):
            _raise_corrupt()
        values = data[shape.index].aligned()
        return [values[start] for start, _ in leaf_ranges]
    if isinstance(shape, _StructShape):
        names = [name for name, _ in shape.fields]
        columns = [_assemble(child, ranges, data) for _, child in shape.fields]
        rows: list[Any] = [
            dict(zip(names, row, strict=True)) for row in zip(*columns, strict=True)
        ]
        if shape.present is not None:
            is_present = _struct_presence(shape.leaves, shape.present, ranges, data)
            rows = [
                row if row_present else None
                for row, row_present in zip(rows, is_present, strict=True)
            ]
        return rows
    element_ranges: Ranges = {}
    lengths: list[int | None] | None = None
    for index in shape.leaves:
        element_ranges[index], leaf_lengths = _split_lists(
            data[index], ranges[index], shape
        )
        if lengths is None:
            lengths = leaf_lengths
        elif leaf_lengths != lengths:  # all leaves must agree on every list's length
            _raise_corrupt()
    elements = iter(_assemble(shape.element, element_ranges, data))
    return [None if n is None else list(islice(elements, n)) for n in lengths or []]


def _column_values(shape: Shape, data: dict[int, _LeafData], n_rows: int) -> list[Any]:
    """A top-level column's values, in one row group."""
    if isinstance(shape, _LeafShape):  # flat, the common case: skip the assembly
        leaf = data[shape.index]
        values = _with_nulls(leaf.values, leaf.defs)
        if len(values) != n_rows:
            _raise_corrupt()
        return values
    ranges = {index: _row_ranges(data[index], n_rows) for index in shape.leaves}
    return _assemble(shape, ranges, data)


def _decode_values(
    page: bytes,
    pos: int,
    encoding: int,
    count: int,
    *,
    column: _Column,
    dictionary: list[Any] | None,
) -> list[Any]:
    if count == 0:  # e.g. an all-null page, which may have no value bytes at all
        return []
    physical = column.physical
    if encoding == PLAIN:
        return _decode_plain_values(page, pos, count, column)
    if encoding in {PLAIN_DICTIONARY, RLE_DICTIONARY}:
        if dictionary is None or (bit_width := page[pos]) > 32:
            _raise_corrupt()
        indices = _decode_rle_bitpacked(page, pos + 1, len(page), bit_width, count)
        return list(map(dictionary.__getitem__, indices))
    if encoding == RLE and physical == BOOLEAN:
        length = int.from_bytes(page[pos : pos + 4], "little")
        bits = _decode_rle_bitpacked(page, pos + 4, pos + 4 + length, 1, count)
        return [bool(bit) for bit in bits]
    if encoding == BYTE_STREAM_SPLIT and physical not in {BOOLEAN, BYTE_ARRAY}:
        width = _value_width(physical, column.type_length)
        data = _byte_stream_split(page, pos, count, width)
        return _convert(_decode_plain(data, 0, physical, count, width), column)
    if encoding == DELTA_BINARY_PACKED and physical in _BITS:
        values, _ = _decode_delta_binary_packed(page, pos, _BITS[physical], count)
    elif encoding == DELTA_LENGTH_BYTE_ARRAY and physical == BYTE_ARRAY:
        values, _ = _decode_delta_length_byte_arrays(page, pos, count)
    elif encoding == DELTA_BYTE_ARRAY and physical in {BYTE_ARRAY, FIXED_LEN_BYTE_ARRAY}:
        values = _decode_delta_byte_arrays(page, pos, count)
    else:
        name = _ENCODING_NAMES.get(encoding, encoding)
        msg = f"Parquet encoding {name} is not supported for column {column.name!r}."
        raise NotImplementedError(msg)
    return _convert(values, column)


def _with_nulls(values: list[Any], levels: list[int] | None) -> list[Any]:
    """Spread the non-null `values` over the definition `levels`, filling in nulls."""
    if levels is None or len(values) == len(levels):
        return values
    it = iter(values)
    return [next(it) if level else None for level in levels]


Levels: TypeAlias = "list[int] | None"
PageData: TypeAlias = "tuple[Levels, Levels, list[Any]]"


def _read_levels_v1(
    page: bytes, pos: int, max_level: int, count: int, encoding: int
) -> tuple[list[int] | None, int]:
    """Read the 4-byte length-prefixed levels at `pos`, if there are any."""
    if not max_level:
        return None, pos
    if encoding != RLE:
        msg = "Only RLE-encoded repetition and definition levels are supported."
        raise NotImplementedError(msg)
    _check_available(page, pos + 4)
    length = int.from_bytes(page[pos : pos + 4], "little")
    end = pos + 4 + length
    _check_available(page, end)
    return _read_levels(page, pos + 4, end, max_level, count), end


def _read_levels(
    buf: bytes, start: int, end: int, max_level: int, count: int
) -> list[int]:
    levels = _decode_rle_bitpacked(buf, start, end, max_level.bit_length(), count)
    # An RLE run's value is a whole byte, so may exceed what `bit_length` bits hold.
    if max(levels, default=0) > max_level:
        _raise_corrupt()
    return levels


def _read_data_page_v1(
    page: bytes, header: dict[int, Any], *, column: _Column, dictionary: list[Any] | None
) -> PageData:
    count, encoding = header[1], header[2]
    if count == 0:
        return [], [], []
    reps, pos = _read_levels_v1(page, 0, column.max_rep, count, header.get(4, RLE))
    defs, pos = _read_levels_v1(page, pos, column.max_def, count, header.get(3, RLE))
    n_values = count if defs is None else defs.count(column.max_def)
    values = _decode_values(
        page, pos, encoding, n_values, column=column, dictionary=dictionary
    )
    return defs, reps, values


def _read_data_page_v2(
    raw: bytes,
    header: dict[int, Any],
    codec: int,
    size: int,
    *,
    column: _Column,
    dictionary: list[Any] | None,
) -> PageData:
    # Unlike V1, levels come first and uncompressed, and only the values may be compressed.
    count, encoding, def_length, rep_length = header[1], header[4], header[5], header[6]
    levels_end = rep_length + def_length
    if def_length < 0 or rep_length < 0 or levels_end > min(len(raw), size):
        _raise_corrupt()
    reps = (
        _read_levels(raw, 0, rep_length, column.max_rep, count)
        if column.max_rep
        else None
    )
    defs = (
        _read_levels(raw, rep_length, levels_end, column.max_def, count)
        if column.max_def
        else None
    )
    body = raw[levels_end:]
    # An all-null page may have an empty body, which e.g. snappy can't decompress.
    compressed = bool(body) and header.get(7, True)
    page = _decompress(codec, body, size - levels_end) if compressed else body
    n_values = count if defs is None else defs.count(column.max_def)
    values = _decode_values(
        page, 0, encoding, n_values, column=column, dictionary=dictionary
    )
    return defs, reps, values


def _chunk_range(meta: dict[int, Any]) -> tuple[int, int]:
    """The `(file offset, size)` of a column chunk's pages."""
    # Dictionary page, if any, else first data page.
    start: int = meta.get(11) or meta[9]
    size: int = meta[7]
    if start < 0 or size < 0:
        _raise_corrupt()
    return start, size


def _pages(
    buf: bytes, pos: int, end: int, column: _Column
) -> Iterator[tuple[dict[int, Any], bytes]]:
    """Each page's header and (still compressed) bytes, between `pos` and `end`."""
    while pos < end:
        r = _Reader(buf, pos)
        header = _read_struct(r)
        if header[2] < 0 or header[3] < 0 or r.pos + header[3] > end:
            _raise_corrupt()
        pos = r.pos + header[3]
        raw = buf[r.pos : pos]
        if (crc := header.get(4)) is not None and zlib.crc32(raw) != crc & 0xFFFFFFFF:
            msg = f"Parquet page checksum mismatch in column {column.name!r}, the file may be corrupt."
            raise ValueError(msg)
        yield header, raw


def _read_column_chunk(
    buf: bytes, meta: dict[int, Any], column: _Column, *, offset: int = 0
) -> _LeafData:
    """Decode a column chunk from `buf`, which starts at file position `offset`."""
    codec: int = meta[4]
    n_expected: int = meta[5]
    start, size = _chunk_range(meta)
    if start < offset or start - offset + size > len(buf):
        _raise_corrupt()
    dictionary: list[Any] | None = None
    defs: list[int] = []
    reps: list[int] = []
    values: list[Any] = []
    n_levels = 0
    for header, raw in _pages(buf, start - offset, start - offset + size, column):
        page_size = header[2]
        if header[1] == DICTIONARY_PAGE:
            dictionary = _decode_plain_values(
                _decompress(codec, raw, page_size), 0, header[7][1], column
            )
            continue
        if header[1] == DATA_PAGE:
            page_header = header[5]
            read_page = partial(
                _read_data_page_v1, _decompress(codec, raw, page_size), page_header
            )
        elif header[1] == DATA_PAGE_V2:
            page_header = header[8]
            read_page = partial(_read_data_page_v2, raw, page_header, codec, page_size)
        else:  # e.g. index pages: no data
            continue
        if not 0 <= page_header[1] <= n_expected - n_levels:
            _raise_corrupt()
        n_levels += page_header[1]
        page_defs, page_reps, page_values = read_page(
            column=column, dictionary=dictionary
        )
        defs += page_defs or ()
        reps += page_reps or ()
        values += page_values
    if n_levels != n_expected:
        _raise_corrupt()
    return _LeafData(
        defs if column.max_def else None,
        reps if column.max_rep else None,
        values,
        column.max_def,
    )


def _check_magic(head: bytes, tail: bytes) -> None:
    if head != MAGIC or tail != MAGIC:
        msg = "Not a Parquet file (or an encrypted one, which is not supported)."
        raise ValueError(msg)


def _read_footer(stream: BinaryIO) -> dict[int, Any]:
    head = stream.read(4)
    file_size = stream.seek(0, 2)
    if file_size < 12:
        _check_magic(head, b"")
    stream.seek(-8, 2)
    tail = stream.read(8)
    _check_magic(head, tail[4:])
    metadata_length = int.from_bytes(tail[:4], "little")
    if metadata_length > file_size - 12:
        _raise_corrupt()
    stream.seek(-8 - metadata_length, 2)
    return _read_struct(_Reader(stream.read(metadata_length)))


def _buffer_footer(buf: bytes) -> dict[int, Any]:
    _check_magic(buf[:4], buf[-4:] if len(buf) >= 12 else b"")
    metadata_length = int.from_bytes(buf[-8:-4], "little")
    if metadata_length > len(buf) - 12:
        _raise_corrupt()
    return _read_struct(_Reader(buf, len(buf) - 8 - metadata_length))


@contextmanager
def _corruption_as_value_error() -> Iterator[None]:
    try:
        yield
    except _CORRUPTION_ERRORS as exc:
        raise ValueError(_CORRUPT_MESSAGE) from exc


def read_parquet(
    source: str | bytes | BinaryIO, columns: Sequence[str] | None = None
) -> dict[str, list[Any]]:
    """Read a Parquet file into a dict of lists, optionally only some `columns`."""
    if isinstance(source, str):
        # Read only the footer and the wanted column chunks, not the whole file.
        with open(source, "rb") as stream:  # noqa: PTH123

            def read_chunk(meta: dict[int, Any], column: _Column) -> _LeafData:
                start, size = _chunk_range(meta)
                stream.seek(start)
                buf = stream.read(size)
                return _read_column_chunk(buf, meta, column, offset=start)

            with _corruption_as_value_error():
                return _read_parquet(_read_footer(stream), columns, read_chunk)
    buf = source if isinstance(source, bytes) else source.read()
    with _corruption_as_value_error():
        return _read_parquet(
            _buffer_footer(buf),
            columns,
            lambda meta, column: _read_column_chunk(buf, meta, column),
        )


def _read_parquet(
    metadata: dict[int, Any],
    columns: Sequence[str] | None,
    read_chunk: Callable[[dict[int, Any], _Column], _LeafData],
) -> dict[str, list[Any]]:
    schema = _Schema(metadata[2])
    names = [name for name, _ in schema.fields]
    check_column_names_are_unique(names)
    if columns is not None and (missing := set(columns).difference(names)):
        msg = f"Columns not found in the Parquet file: {sorted(missing)}."
        raise ValueError(msg)
    shapes = dict(schema.fields)
    wanted = dict.fromkeys(names if columns is None else columns)
    selected = [(name, shapes[name]) for name in wanted]
    result: dict[str, list[Any]] = {name: [] for name, _ in selected}
    row_groups = metadata.get(4, [])
    # Old writers (e.g. parquet-rs 0.3) left the file's row count at 0.
    if metadata[3] and sum(row_group[3] for row_group in row_groups) != metadata[3]:
        _raise_corrupt()
    for row_group in row_groups:
        chunks, n_rows = row_group[1], row_group[3]
        if len(chunks) != len(schema.leaves):
            _raise_corrupt()
        for name, shape in selected:
            data: dict[int, _LeafData] = {}
            for index in shape.leaves:
                if 1 in chunks[index]:
                    msg = "Parquet column chunks stored in other files are not supported."
                    raise NotImplementedError(msg)
                data[index] = read_chunk(chunks[index][3], schema.leaves[index])
            result[name] += _column_values(shape, data, n_rows)
    return result
