"""A minimal, pure-Python Parquet reader which decodes straight into a `dict[str, list]`.

Supported:

- flat (non-nested) columns, required or optional;
- data pages V1 and V2, and dictionary pages;
- `PLAIN`, `PLAIN_DICTIONARY`/`RLE_DICTIONARY` and (for booleans) `RLE` encodings;
- uncompressed, snappy and gzip compression, and zstd on Python 3.14+ (`compression.zstd`);
- strings, enums, JSON, decimals, dates, times, timestamps, and (un)signed integers.

Anything else raises `NotImplementedError`.

The format is specified at https://github.com/apache/parquet-format, and the metadata is
serialized with Thrift's compact protocol. Thrift structs are decoded generically, into
dicts keyed by field id, and only the fields this reader needs are looked up.
"""

from __future__ import annotations

import datetime as dt
import struct
import sys
import zlib
from decimal import Decimal
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
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

            return zstd.decompress(data)
        msg = (
            "Reading zstd-compressed Parquet needs Python 3.14+. On older Python "
            "versions, read it with pyarrow and convert: "
            "`pz.from_arrow(pyarrow.parquet.read_table(path))`."
        )
        raise NotImplementedError(msg)
    msg = f"Parquet compression codec {_CODEC_NAMES.get(codec, codec)} is not supported."
    raise NotImplementedError(msg)


def _decode_rle_bitpacked(
    buf: bytes, pos: int, end: int, bit_width: int, count: int
) -> list[int]:
    """Decode `count` values of the RLE / bit-packing hybrid encoding."""
    if bit_width == 0:
        return [0] * count
    out: list[int] = []
    byte_width = (bit_width + 7) // 8
    mask = (1 << bit_width) - 1
    while len(out) < count and pos < end:
        r = _Reader(buf, pos)
        header = r.varint()
        pos = r.pos
        if header & 1:  # bit-packed groups of 8 values, `bit_width` bytes per group
            n_groups = header >> 1
            # Unpack a few groups at a time, as shifting one huge int is quadratic.
            for first in range(0, n_groups, 64):
                groups = min(64, n_groups - first)
                n_bytes = groups * bit_width
                packed = int.from_bytes(buf[pos : pos + n_bytes], "little")
                pos += n_bytes
                out.extend(
                    (packed >> s) & mask
                    for s in range(0, groups * 8 * bit_width, bit_width)
                )
        else:  # a run of one repeated value
            value = int.from_bytes(buf[pos : pos + byte_width], "little")
            pos += byte_width
            out += [value] * (header >> 1)
    del out[count:]
    return out


def _decode_plain(
    buf: bytes, pos: int, physical: int, count: int, type_length: int | None
) -> list[Any]:
    if fmt := _STRUCT_FORMATS.get(physical):
        return list(struct.unpack_from(f"<{count}{fmt}", buf, pos))
    if physical == BYTE_ARRAY:
        out = []
        for _ in range(count):
            length = int.from_bytes(buf[pos : pos + 4], "little")
            pos += 4
            out.append(buf[pos : pos + length])
            pos += length
        return out
    if physical == BOOLEAN:
        return [bool((buf[pos + i // 8] >> (i % 8)) & 1) for i in range(count)]
    width = 12 if physical == INT96 else type_length
    assert width is not None  # noqa: S101
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
        if physical in {BYTE_ARRAY, FIXED_LEN_BYTE_ARRAY}:
            return lambda b: Decimal(int.from_bytes(b, "big", signed=True)).scaleb(-scale)
        return lambda v: Decimal(v).scaleb(-scale)
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
        values = _decode_plain(page, pos, column.physical, count, column.type_length)
        return list(map(column.convert, values)) if column.convert else values
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
    page = _decompress(codec, body) if header.get(7, True) else body
    values = _decode_values(
        page, 0, encoding, count - n_nulls, column=column, dictionary=dictionary
    )
    return _with_nulls(values, levels)


def _read_column_chunk(buf: bytes, meta: dict[int, Any], column: _Column) -> list[Any]:
    codec: int = meta[4]
    start: int = meta.get(11) or meta[9]  # dictionary page, if any, else first data page
    pos, end = start, start + meta[7]
    dictionary: list[Any] | None = None
    out: list[Any] = []
    while pos < end:
        r = _Reader(buf, pos)
        header = _read_struct(r)
        page_type, compressed_size = header[1], header[3]
        raw = buf[r.pos : r.pos + compressed_size]
        pos = r.pos + compressed_size
        if page_type == DICTIONARY_PAGE:
            page = _decompress(codec, raw)
            count = header[7][1]
            values = _decode_plain(page, 0, column.physical, count, column.type_length)
            dictionary = list(map(column.convert, values)) if column.convert else values
        elif page_type == DATA_PAGE:
            page = _decompress(codec, raw)
            out += _read_data_page_v1(page, header[5], column, dictionary)
        elif page_type == DATA_PAGE_V2:
            out += _read_data_page_v2(raw, header[8], codec, column, dictionary)
        # Other page types (index pages) carry no data.
    return out


def read_parquet(
    source: str | BinaryIO, columns: Sequence[str] | None = None
) -> dict[str, list[Any]]:
    """Read a flat Parquet file into a dict of lists, optionally only some `columns`."""
    if isinstance(source, str):
        with open(source, "rb") as stream:  # noqa: PTH123
            buf = stream.read()
    else:
        buf = source.read()
    if buf[:4] != MAGIC or buf[-4:] != MAGIC:
        msg = "Not a Parquet file (or an encrypted one, which is not supported)."
        raise ValueError(msg)
    metadata_length = int.from_bytes(buf[-8:-4], "little")
    metadata = _read_struct(_Reader(buf, len(buf) - 8 - metadata_length))
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
            result[column.name] += _read_column_chunk(buf, chunk[3], column)
    return result
