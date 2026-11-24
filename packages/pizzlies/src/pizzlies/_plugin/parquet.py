"""A minimal, pure-Python Parquet reader which decodes straight into a `dict[str, list]`.

Supported:

- flat (non-nested) columns, required or optional;
- data pages V1 and V2, and dictionary pages;
- `PLAIN`, `PLAIN_DICTIONARY`/`RLE_DICTIONARY` and (for booleans) `RLE` encodings;
- uncompressed, snappy and gzip compression, and zstd with `compression.zstd` (Python
  3.14+) or else the optional `backports.zstd`;
- strings, enums, JSON, decimals, dates, times, timestamps, and (un)signed integers.

Anything else raises `NotImplementedError`.

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
from decimal import Decimal
from functools import cache
from itertools import chain
from typing import TYPE_CHECKING, Any, NoReturn

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence
    from typing import BinaryIO

    Converter = Callable[[Any], Any]

MAGIC = b"PAR1"

# `Type`
BOOLEAN, INT32, INT64, INT96, FLOAT, DOUBLE, BYTE_ARRAY, FIXED_LEN_BYTE_ARRAY = range(8)
_STRUCT_FORMATS = {INT32: "i", INT64: "q", FLOAT: "f", DOUBLE: "d"}

# `FieldRepetitionType`
OPTIONAL, REPEATED = 1, 2

# `PageType`
DATA_PAGE, DICTIONARY_PAGE, DATA_PAGE_V2 = 0, 2, 3

# `Encoding`
PLAIN, PLAIN_DICTIONARY, RLE, RLE_DICTIONARY = 0, 2, 3, 8
_ENCODING_NAMES = {
    4: "BIT_PACKED",
    5: "DELTA_BINARY_PACKED",
    6: "DELTA_LENGTH_BYTE_ARRAY",
    7: "DELTA_BYTE_ARRAY",
    9: "BYTE_STREAM_SPLIT",
}

# `CompressionCodec`
UNCOMPRESSED, SNAPPY, GZIP, ZSTD = 0, 1, 2, 6
_CODEC_NAMES = {3: "LZO", 4: "BROTLI", 5: "LZ4", 7: "LZ4_RAW"}

# `ConvertedType`, for files which predate logical types
_UTF8, _ENUM, _DECIMAL, _DATE, _TIME_MILLIS, _TIME_MICROS = 0, 4, 5, 6, 7, 8
_TIMESTAMP_MILLIS, _TIMESTAMP_MICROS, _JSON = 9, 10, 19
_UNSIGNED_BITS = {11: 8, 12: 16, 13: 32, 14: 64}

# `LogicalType` union members
_L_STRING, _L_ENUM, _L_DECIMAL, _L_DATE, _L_TIME, _L_TIMESTAMP = 1, 4, 5, 6, 7, 8
_L_INTEGER, _L_JSON = 10, 12
_L_UNSUPPORTED = {2: "MAP", 3: "LIST", 15: "FLOAT16", 16: "VARIANT"}

# `TimeUnit` union members, to a divisor which turns the stored value into microseconds
_MILLIS, _MICROS, _NANOS = 1, 2, 3

_EPOCH_ORDINAL = dt.date(1970, 1, 1).toordinal()
_JULIAN_EPOCH = 2_440_588  # Julian day number of 1970-01-01, for INT96 timestamps


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
    msg = f"Unexpected Thrift compact type {kind}, the Parquet file may be corrupt."
    raise ValueError(msg)


def _read_struct(r: _Reader) -> dict[int, Any]:
    fields: dict[int, Any] = {}
    field_id = 0
    while header := r.byte():
        delta = header >> 4
        field_id = field_id + delta if delta else r.zigzag()
        fields[field_id] = _read_value(r, header & 0x0F)
    return fields


def _snappy_decompress(data: bytes) -> bytes:
    r = _Reader(data)
    r.varint()  # uncompressed length
    pos, end, out = r.pos, len(data), bytearray()
    while pos < end:
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
        start = len(out) - offset
        if offset >= length:
            out += out[start : start + length]
        else:  # the copy overlaps what it produces, so it repeats the last `offset` bytes
            out += (out[start:] * (length // offset + 1))[:length]
    return bytes(out)


def _decompress(codec: int, data: bytes) -> bytes:
    if codec == UNCOMPRESSED:
        return data
    if codec == SNAPPY:
        return _snappy_decompress(data)
    if codec == GZIP:
        return zlib.decompress(data, wbits=zlib.MAX_WBITS | 32)
    if codec == ZSTD:
        if sys.version_info >= (3, 14):
            from compression import zstd
        else:
            try:  # optional: `pip install pizzlies[zstd]`
                from backports import zstd  # type: ignore[import-not-found,unused-ignore]  # pyright: ignore[reportMissingImports]
            except ImportError:
                msg = (
                    "Reading zstd-compressed Parquet needs Python 3.14+, or the "
                    "`backports.zstd` package (`pip install pizzlies[zstd]`)."
                )
                raise NotImplementedError(msg) from None
        result: bytes = zstd.decompress(data)
        return result
    msg = f"Parquet compression codec {_CODEC_NAMES.get(codec, codec)} is not supported."
    raise NotImplementedError(msg)


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


def _raise_corrupt() -> NoReturn:
    msg = "Truncated or invalid Parquet data, the file may be corrupt."
    raise ValueError(msg)


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


def _decode_plain_values(page: bytes, pos: int, count: int, column: _Column) -> list[Any]:
    """`_decode_plain`, converted to Python values."""
    if column.physical == BYTE_ARRAY and column.convert is bytes.decode:
        return _decode_byte_arrays(page, pos, count, utf8=True)
    values = _decode_plain(page, pos, column.physical, count, column.type_length)
    return list(map(column.convert, values)) if column.convert else values


def _decode_plain(
    buf: bytes, pos: int, physical: int, count: int, type_length: int | None
) -> list[Any]:
    if physical == BYTE_ARRAY:
        return _decode_byte_arrays(buf, pos, count, utf8=False)
    if fmt := _STRUCT_FORMATS.get(physical):
        fmt = f"<{count}{fmt}"
        _check_available(buf, pos + struct.calcsize(fmt))
        return list(struct.unpack_from(fmt, buf, pos))
    if physical == BOOLEAN:
        _check_available(buf, pos + (count + 7) // 8)
        return [bool((buf[pos + i // 8] >> (i % 8)) & 1) for i in range(count)]
    width = 12 if physical == INT96 else type_length
    assert width is not None  # noqa: S101
    _check_available(buf, pos + count * width)
    return [buf[pos + i * width : pos + (i + 1) * width] for i in range(count)]


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
    if _L_DECIMAL in logical or converted == _DECIMAL:
        scale = (
            logical[_L_DECIMAL].get(1, 0) if _L_DECIMAL in logical else element.get(7, 0)
        )
        # Built from a string, as `scaleb` would round to the context's precision.
        if physical in {BYTE_ARRAY, FIXED_LEN_BYTE_ARRAY}:
            return lambda b: Decimal(f"{int.from_bytes(b, 'big', signed=True)}E-{scale}")
        return lambda v: Decimal(f"{v}E-{scale}")
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
    __slots__ = ("convert", "max_def", "name", "physical", "type_length")

    def __init__(self, element: dict[int, Any]) -> None:
        self.name: str = element[4].decode()
        self.physical: int = element[1]
        self.type_length: int | None = element.get(2)
        self.max_def = int(element.get(3) == OPTIONAL)
        self.convert = _converter(element)


def _columns(schema: list[dict[int, Any]]) -> list[_Column]:
    root, *elements = schema
    if any(5 in el or el.get(3) == REPEATED for el in elements) or root.get(5) != len(
        elements
    ):
        msg = "Nested Parquet columns (lists, structs, maps) are not supported."
        raise NotImplementedError(msg)
    return [_Column(el) for el in elements]


def _decode_values(
    page: bytes,
    pos: int,
    encoding: int,
    count: int,
    *,
    column: _Column,
    dictionary: list[Any] | None,
) -> list[Any]:
    if encoding == PLAIN:
        return _decode_plain_values(page, pos, count, column)
    if encoding in {PLAIN_DICTIONARY, RLE_DICTIONARY}:
        assert dictionary is not None  # noqa: S101
        indices = _decode_rle_bitpacked(page, pos + 1, len(page), page[pos], count)
        return list(map(dictionary.__getitem__, indices))
    if encoding == RLE and column.physical == BOOLEAN:
        length = int.from_bytes(page[pos : pos + 4], "little")
        bits = _decode_rle_bitpacked(page, pos + 4, pos + 4 + length, 1, count)
        return [bool(bit) for bit in bits]
    msg = f"Parquet encoding {_ENCODING_NAMES.get(encoding, encoding)} is not supported."
    raise NotImplementedError(msg)


def _with_nulls(values: list[Any], levels: list[int] | None) -> list[Any]:
    """Spread the non-null `values` over the definition `levels`, filling in nulls."""
    if levels is None or len(values) == len(levels):
        return values
    it = iter(values)
    return [next(it) if level else None for level in levels]


def _read_data_page_v1(
    page: bytes, header: dict[int, Any], column: _Column, dictionary: list[Any] | None
) -> list[Any]:
    count, encoding, level_encoding = header[1], header[2], header[3]
    if not column.max_def:
        return _decode_values(
            page, 0, encoding, count, column=column, dictionary=dictionary
        )
    if level_encoding != RLE:
        msg = "Only RLE-encoded definition levels are supported."
        raise NotImplementedError(msg)
    length = int.from_bytes(page[:4], "little")
    levels = _decode_rle_bitpacked(page, 4, 4 + length, 1, count)
    values = _decode_values(
        page, 4 + length, encoding, sum(levels), column=column, dictionary=dictionary
    )
    return _with_nulls(values, levels)


def _read_data_page_v2(
    raw: bytes,
    header: dict[int, Any],
    codec: int,
    column: _Column,
    dictionary: list[Any] | None,
) -> list[Any]:
    # Unlike V1, levels come first and uncompressed, and only the values may be compressed.
    count, n_nulls, encoding = header[1], header[2], header[4]
    rep_length = header[6]
    levels_end = rep_length + header[5]
    levels = (
        _decode_rle_bitpacked(raw, rep_length, levels_end, 1, count)
        if column.max_def
        else None
    )
    body = raw[levels_end:]
    # An all-null page may have an empty body, which e.g. snappy can't decompress.
    page = _decompress(codec, body) if body and header.get(7, True) else body
    values = _decode_values(
        page, 0, encoding, count - n_nulls, column=column, dictionary=dictionary
    )
    return _with_nulls(values, levels)


def _chunk_range(meta: dict[int, Any]) -> tuple[int, int]:
    """The `(file offset, size)` of a column chunk's pages."""
    # Dictionary page, if any, else first data page.
    start: int = meta.get(11) or meta[9]
    size: int = meta[7]
    if start < 0 or size < 0:
        _raise_corrupt()
    return start, size


def _read_column_chunk(
    buf: bytes, meta: dict[int, Any], column: _Column, *, offset: int = 0
) -> list[Any]:
    """Decode a column chunk from `buf`, which starts at file position `offset`."""
    codec: int = meta[4]
    start, size = _chunk_range(meta)
    pos = start - offset
    end = pos + size
    if end > len(buf):
        _raise_corrupt()
    dictionary: list[Any] | None = None
    out: list[Any] = []
    while pos < end:
        r = _Reader(buf, pos)
        header = _read_struct(r)
        page_type, compressed_size = header[1], header[3]
        if compressed_size < 0 or r.pos + compressed_size > end:
            _raise_corrupt()
        raw = buf[r.pos : r.pos + compressed_size]
        pos = r.pos + compressed_size
        if page_type == DICTIONARY_PAGE:
            page = _decompress(codec, raw)
            dictionary = _decode_plain_values(page, 0, header[7][1], column)
        elif page_type == DATA_PAGE:
            page = _decompress(codec, raw)
            out += _read_data_page_v1(page, header[5], column, dictionary)
        elif page_type == DATA_PAGE_V2:
            out += _read_data_page_v2(raw, header[8], codec, column, dictionary)
        # Other page types (index pages) carry no data.
    return out


def _check_magic(head: bytes, tail: bytes) -> None:
    if head != MAGIC or tail != MAGIC:
        msg = "Not a Parquet file (or an encrypted one, which is not supported)."
        raise ValueError(msg)


def _read_footer(stream: BinaryIO) -> dict[int, Any]:
    head = stream.read(4)
    stream.seek(-8, 2)
    tail = stream.read(8)
    _check_magic(head, tail[4:])
    metadata_length = int.from_bytes(tail[:4], "little")
    stream.seek(-8 - metadata_length, 2)
    return _read_struct(_Reader(stream.read(metadata_length)))


def _buffer_footer(buf: bytes) -> dict[int, Any]:
    _check_magic(buf[:4], buf[-4:])
    metadata_length = int.from_bytes(buf[-8:-4], "little")
    return _read_struct(_Reader(buf, len(buf) - 8 - metadata_length))


def read_parquet(
    source: str | bytes | BinaryIO, columns: Sequence[str] | None = None
) -> dict[str, list[Any]]:
    """Read a flat Parquet file into a dict of lists, optionally only some `columns`."""
    if isinstance(source, str):
        # Read only the footer and the wanted column chunks, not the whole file.
        with open(source, "rb") as stream:  # noqa: PTH123

            def read_chunk(meta: dict[int, Any], column: _Column) -> list[Any]:
                start, size = _chunk_range(meta)
                stream.seek(start)
                buf = stream.read(size)
                return _read_column_chunk(buf, meta, column, offset=start)

            return _read_parquet(_read_footer(stream), columns, read_chunk)
    buf = source if isinstance(source, bytes) else source.read()
    return _read_parquet(
        _buffer_footer(buf),
        columns,
        lambda meta, column: _read_column_chunk(buf, meta, column),
    )


def _read_parquet(
    metadata: dict[int, Any],
    columns: Sequence[str] | None,
    read_chunk: Callable[[dict[int, Any], _Column], list[Any]],
) -> dict[str, list[Any]]:
    all_columns = _columns(metadata[2])
    names = [column.name for column in all_columns]
    if columns is not None and (missing := set(columns).difference(names)):
        msg = f"Columns not found in the Parquet file: {sorted(missing)}."
        raise ValueError(msg)
    wanted = set(names if columns is None else columns)
    result: dict[str, list[Any]] = {
        name: [] for name in (names if columns is None else columns)
    }
    for row_group in metadata.get(4, []):
        for column, chunk in zip(all_columns, row_group[1], strict=True):
            if column.name not in wanted:
                continue
            if 1 in chunk:
                msg = "Parquet column chunks stored in other files are not supported."
                raise NotImplementedError(msg)
            result[column.name] += read_chunk(chunk[3], column)
    return result
