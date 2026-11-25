from __future__ import annotations

import datetime as dt
import random
import string
import sys
import zlib
from contextlib import suppress
from decimal import Decimal
from importlib.util import find_spec
from io import BytesIO
from typing import TYPE_CHECKING, Any

import pizzlies as pz
import pytest
from pizzlies._plugin.parquet import (
    BOOLEAN,
    BYTE_ARRAY,
    DELTA_BINARY_PACKED,
    DELTA_BYTE_ARRAY,
    DELTA_LENGTH_BYTE_ARRAY,
    FIXED_LEN_BYTE_ARRAY,
    GZIP,
    INT32,
    INT64,
    LZ4,
    OPTIONAL,
    PLAIN,
    REPEATED,
    RLE_DICTIONARY,
    SNAPPY,
    _buffer_footer,
    _Column,
    _column_values,
    _decode_plain,
    _decode_plain_values,
    _decode_rle_bitpacked,
    _decode_values,
    _decompress,
    _LeafData,
    _ListShape,
    _read_column_chunk,
    _read_data_page_v2,
    _read_parquet,
    _Schema,
    read_parquet,
)

if TYPE_CHECKING:
    from pathlib import Path

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

CODECS = ["none", "snappy", "gzip", "lz4", "zstd"]
UTC = dt.timezone.utc


def _has_zstd() -> bool:
    # The stdlib has zstd from Python 3.14, and the `backports.zstd` package before that.
    if sys.version_info >= (3, 14):
        return True
    try:
        return find_spec("backports.zstd") is not None
    except ModuleNotFoundError:  # `backports` itself isn't installed
        return False


HAS_ZSTD = _has_zstd()


def _codec_param(codec: str) -> Any:
    marks = []
    if codec == "zstd" and not HAS_ZSTD:
        marks.append(pytest.mark.skip(reason="needs Python 3.14+ or backports.zstd"))
    return pytest.param(codec, marks=marks)


def _table() -> Any:
    rng = random.Random(0)  # noqa: S311
    n = 300
    words = [
        "".join(rng.choices(string.ascii_letters, k=rng.randint(0, 80))) for _ in range(n)
    ]

    def nullify(values: list[Any]) -> list[Any]:
        return [None if i % 7 == 3 else v for i, v in enumerate(values)]

    columns: dict[str, Any] = {
        "i8": pa.array(nullify([rng.randint(-128, 127) for _ in range(n)]), pa.int8()),
        "i16": pa.array([rng.randint(-(2**15), 2**15 - 1) for _ in range(n)], pa.int16()),
        "i32": pa.array(nullify([rng.randint(-(2**31), 2**31 - 1) for _ in range(n)])),
        "i64": pa.array([rng.randint(-(2**63), 2**63 - 1) for _ in range(n)], pa.int64()),
        "u8": pa.array([rng.randint(0, 255) for _ in range(n)], pa.uint8()),
        "u32": pa.array([rng.randint(0, 2**32 - 1) for _ in range(n)], pa.uint32()),
        "u64": pa.array([rng.randint(0, 2**64 - 1) for _ in range(n)], pa.uint64()),
        "f32": pa.array([float(i) / 4 for i in range(n)], pa.float32()),
        "f64": pa.array(nullify([rng.random() for _ in range(n)])),
        "bool": pa.array(nullify([rng.random() < 0.5 for _ in range(n)])),
        "str": pa.array(nullify(words)),
        "low_cardinality": pa.array(nullify([rng.choice("abc") for _ in range(n)])),
        "repetitive": pa.array(["ab" * rng.randint(0, 50) for _ in range(n)]),
        "binary": pa.array([w.encode() for w in words], pa.binary()),
        "date": pa.array(
            nullify([dt.date(2000, 1, 1) + dt.timedelta(days=i * 13) for i in range(n)])
        ),
        "time_ms": pa.array(
            [dt.time(1, 2, 3, (i * 1000) % 1_000_000) for i in range(n)], pa.time32("ms")
        ),
        "time_us": pa.array([dt.time(1, 2, 3, i) for i in range(n)], pa.time64("us")),
        "decimal": pa.array(
            [Decimal(rng.randint(-(10**9), 10**9)).scaleb(-3) for _ in range(n)],
            pa.decimal128(12, 3),
        ),
    }
    base = dt.datetime(2020, 1, 1, 12)
    stamps = nullify(
        [base + dt.timedelta(seconds=i * 3601, microseconds=i) for i in range(n)]
    )
    for unit in ("ms", "us", "ns"):
        values = (
            stamps if unit != "ms" else [s and s.replace(microsecond=0) for s in stamps]
        )
        columns[f"ts_{unit}"] = pa.array(values, pa.timestamp(unit))
    columns["ts_utc"] = pa.array(
        [s and s.replace(tzinfo=UTC) for s in stamps], pa.timestamp("us", "UTC")
    )
    return pa.table(columns)


def _write(table: Any, **kwargs: Any) -> BytesIO:
    buf = BytesIO()
    pq.write_table(table, buf, **kwargs)
    buf.seek(0)
    return buf


@pytest.mark.parametrize("codec", [_codec_param(c) for c in CODECS])
@pytest.mark.parametrize("data_page_version", ["1.0", "2.0"])
@pytest.mark.parametrize("use_dictionary", [True, False])
def test_roundtrip(codec: str, data_page_version: str, use_dictionary: bool) -> None:  # noqa: FBT001
    table = _table()
    buf = _write(
        table,
        compression=codec,
        data_page_version=data_page_version,
        use_dictionary=use_dictionary,
        row_group_size=128,
        data_page_size=512,
    )
    result = read_parquet(buf)
    assert result == table.to_pydict()
    assert all(type(v) is list for v in result.values())


@pytest.mark.parametrize("use_dictionary", [True, False])
@pytest.mark.parametrize(
    "strings",
    [
        ["ascii", "", None, "short"],
        ["héllo", "naïve", None, "日本語", "🐍"],  # non-ASCII: decoded value by value
        ["x" * 200, "y", None],  # a length byte >= 128 makes the data non-ASCII
    ],
)
def test_strings(strings: list[str | None], use_dictionary: bool) -> None:  # noqa: FBT001
    table = pa.table({"s": strings * 50})
    buf = _write(table, use_dictionary=use_dictionary)
    assert read_parquet(buf) == table.to_pydict()


@pytest.mark.parametrize("n_distinct", [3, 1000, 70_000])
def test_mixed_rle_and_bitpacked_runs(n_distinct: int) -> None:
    # Long repeats become RLE runs, varied stretches bit-packed ones, interleaved.
    rng = random.Random(n_distinct)  # noqa: S311
    values: list[int] = []
    while len(values) < 20_000:
        if rng.random() < 0.5:
            values += [rng.randrange(n_distinct)] * rng.randint(8, 100)
        else:
            values += [rng.randrange(n_distinct) for _ in range(rng.randint(1, 50))]
    table = pa.table({"a": values})
    assert read_parquet(_write(table, data_page_size=4096)) == {"a": values}


def test_int96_timestamps() -> None:
    values = [
        dt.datetime(1999, 12, 31, 23, 59, 59, 123456),
        None,
        dt.datetime(1960, 1, 1),
    ]
    table = pa.table({"a": pa.array(values, pa.timestamp("us"))})
    buf = _write(table, use_deprecated_int96_timestamps=True)
    assert read_parquet(buf) == {"a": values}


def test_required_columns() -> None:
    schema = pa.schema(
        [
            pa.field("a", pa.int64(), nullable=False),
            pa.field("b", pa.string(), nullable=False),
        ]
    )
    table = pa.table({"a": [1, 2, 3], "b": ["x", "y", "z"]}, schema=schema)
    assert read_parquet(_write(table)) == {"a": [1, 2, 3], "b": ["x", "y", "z"]}


def test_written_by_polars() -> None:
    pl = pytest.importorskip("polars")
    df = pl.DataFrame(
        {"a": [1, None, 3], "b": ["x", "y", None], "c": [True, None, False]}
    )
    for codec in ("uncompressed", "snappy", "gzip") + (("zstd",) if HAS_ZSTD else ()):
        buf = BytesIO()
        df.write_parquet(buf, compression=codec)
        buf.seek(0)
        assert read_parquet(buf) == df.to_dict(as_series=False)


def test_empty_and_columns(tmp_path: Path) -> None:
    path = tmp_path / "data.parquet"
    pq.write_table(
        pa.table({"a": pa.array([], pa.int64()), "b": pa.array([], pa.string())}), path
    )
    assert read_parquet(str(path)) == {"a": [], "b": []}
    pq.write_table(pa.table({"a": [1, 2], "b": ["x", "y"], "c": [1.0, 2.0]}), path)
    assert read_parquet(str(path), columns=["c", "a"]) == {"c": [1.0, 2.0], "a": [1, 2]}
    with pytest.raises(ValueError, match="not found"):
        read_parquet(str(path), columns=["z"])


def test_public_api(tmp_path: Path) -> None:
    path = tmp_path / "data.parquet"
    pq.write_table(pa.table({"a": [1, 2], "b": ["x", None]}), path)
    df = pz.read_parquet(path)
    assert df.to_native() == {"a": [1, 2], "b": ["x", None]}
    assert pz.read_parquet(str(path), columns=["b"]).to_native() == {"b": ["x", None]}
    assert pz.scan_parquet(path).collect().to_native() == {"a": [1, 2], "b": ["x", None]}
    with pytest.raises(TypeError, match="Unsupported keyword arguments"):
        pz.read_parquet(path, engine="pyarrow")


def test_not_parquet() -> None:
    with pytest.raises(ValueError, match="Not a Parquet file"):
        read_parquet(BytesIO(b"definitely not parquet"))


def test_decimal_beyond_context_precision() -> None:
    values = [Decimal("1234567890123456789012345678901234.5678"), Decimal("-0.0010")]
    for dtype in [pa.decimal128(38, 4), pa.decimal128(9, 4)]:
        data = values if dtype.precision == 38 else [values[1]]
        result = read_parquet(_write(pa.table({"a": pa.array(data, dtype)})))["a"]
        assert result == data
        assert [v.as_tuple() for v in result] == [v.as_tuple() for v in data]


@pytest.mark.parametrize("compressed_size", [-1, 63])
def test_invalid_page_size(compressed_size: int) -> None:
    raw = bytearray(
        _write(
            pa.table({"a": [1]}),
            compression="none",
            use_dictionary=False,
            write_statistics=False,
        ).getvalue()
    )
    # The first page header: three i32 fields, `type`, `uncompressed_page_size`
    # and `compressed_page_size`, each a one-byte zigzag varint here.
    assert raw[4:10:2] == b"\x15\x15\x15"
    assert raw[9] < 128
    raw[9] = compressed_size << 1 if compressed_size >= 0 else -2 * compressed_size - 1
    with pytest.raises(ValueError, match="may be corrupt"):
        read_parquet(bytes(raw))


@pytest.mark.parametrize(
    ("buf", "physical", "count", "type_length"),
    [
        (b"\x00" * 12, INT64, 2, None),
        (b"\x00", BOOLEAN, 9, None),
        (b"\x00" * 7, FIXED_LEN_BYTE_ARRAY, 2, 4),
        (b"\x00" * 8, FIXED_LEN_BYTE_ARRAY, 2, None),  # no `type_length` in the schema
        (b"\x03\x00\x00\x00ab", BYTE_ARRAY, 1, None),
        (b"\x00\x00\x00\x00", BYTE_ARRAY, 2, None),
    ],
)
def test_truncated_plain_values(
    buf: bytes, physical: int, count: int, type_length: int | None
) -> None:
    with pytest.raises(ValueError, match="may be corrupt"):
        _decode_plain(buf, 0, physical, count, type_length)


def test_truncated_or_oversized_rle_runs() -> None:
    assert _decode_rle_bitpacked(b"\x06\x01", 0, 2, 1, 3) == [1, 1, 1]
    with pytest.raises(ValueError, match="may be corrupt"):
        _decode_rle_bitpacked(b"\xfe\xff\xff\xff\x0f\x01", 0, 6, 1, 3)
    with pytest.raises(ValueError, match="may be corrupt"):
        _decode_rle_bitpacked(b"\x04\x01", 0, 2, 1, 3)
    # RLE value byte or bit-packed payload missing, within `end` or past the buffer.
    for buf, end in [(b"\x06", 1), (b"\x06", 5), (b"\x03", 1), (b"\x03", 5)]:
        with pytest.raises(ValueError, match="may be corrupt"):
            _decode_rle_bitpacked(buf, 0, end, 1, 3)


def test_unsupported() -> None:
    with pytest.raises(NotImplementedError, match="BROTLI"):
        read_parquet(_write(pa.table({"a": [1, 2]}), compression="brotli"))
    column = _Column({1: INT32, 4: b"a"})
    with pytest.raises(NotImplementedError, match="BIT_PACKED"):
        _decode_values(b"\x00", 0, 4, 1, column=column, dictionary=None)
    with pytest.raises(NotImplementedError, match="encoding 6"):  # not for integers
        _decode_values(b"\x00", 0, 6, 1, column=column, dictionary=None)


@pytest.mark.skipif(sys.version_info >= (3, 14), reason="the stdlib has zstd")
def test_zstd_without_backport(monkeypatch: pytest.MonkeyPatch) -> None:
    buf = _write(pa.table({"a": [1, 2]}), compression="zstd")
    # Make `from backports import zstd` fail, even if it was imported already.
    monkeypatch.setitem(sys.modules, "backports.zstd", None)
    if (backports := sys.modules.get("backports")) is not None:
        monkeypatch.delattr(backports, "zstd", raising=False)
    with pytest.raises(NotImplementedError, match=r"pizzlies\[zstd\]"):
        read_parquet(buf)


@pytest.mark.parametrize("encoding", [PLAIN, RLE_DICTIONARY])
def test_all_null_v2_page_with_empty_body(encoding: int) -> None:
    # parquet-java can write all-null V2 pages with no (compressed) value bytes at all.
    column = _Column({1: INT64, 3: OPTIONAL, 4: b"a"})
    levels = b"\x06\x00"  # an RLE run of three 0s: three nulls
    header = {1: 3, 2: 3, 4: encoding, 5: len(levels), 6: 0}
    page = _read_data_page_v2(
        levels, header, SNAPPY, len(levels), column=column, dictionary=None
    )
    assert page == ([0, 0, 0], None, [])


def test_dictionary_page_missing() -> None:
    column = _Column({1: INT64, 4: b"a"})
    header = {1: 1, 2: 0, 4: RLE_DICTIONARY, 5: 0, 6: 0, 7: False}
    with pytest.raises(ValueError, match="may be corrupt"):
        _read_data_page_v2(
            b"\x01\x02\x00", header, SNAPPY, 3, column=column, dictionary=None
        )


def test_annotation_on_wrong_physical_type() -> None:
    # A TIMESTAMP_MILLIS converted type on a 12-byte FIXED_LEN_BYTE_ARRAY.
    with pytest.raises(NotImplementedError, match="INT32 or INT64"):
        _Column({1: FIXED_LEN_BYTE_ARRAY, 2: 12, 4: b"a", 6: 9})


@pytest.mark.parametrize(
    "array", [pa.array([0, 2**62], pa.timestamp("us")), pa.array([0, 2**30], pa.date32())]
)
def test_value_out_of_python_range(array: Any) -> None:
    with pytest.raises(ValueError, match="'a' holds a value Python can't represent"):
        read_parquet(_write(pa.table({"a": array})))


def test_gzip_multiple_members() -> None:
    import gzip
    import zlib

    first, second = gzip.compress(b"abc"), gzip.compress(b"def")
    assert _decompress(GZIP, first + second, 6) == b"abcdef"
    assert _decompress(GZIP, zlib.compress(b"abc"), 3) == b"abc"
    with pytest.raises(ValueError, match="may be corrupt"):
        _decompress(GZIP, first + second[:-4], 6)
    with pytest.raises(ValueError, match="may be corrupt"):
        _decompress(GZIP, first + b"garbage", 6)


def test_duplicate_column_names() -> None:
    from narwhals.exceptions import DuplicateError

    buf = _write(pa.table([[1, 2], [10, 20]], names=["a", "a"]))
    with pytest.raises(DuplicateError, match="'a' 2 times"):
        read_parquet(buf)


def test_path_reads_only_selected_column_chunks(tmp_path: Path) -> None:
    path = tmp_path / "data.parquet"
    table = pa.table({"a": [1, 2], "b": ["x", None], "c": [1.5, 2.5]})
    pq.write_table(table, path, row_group_size=1)
    assert read_parquet(str(path), ["c", "a"]) == {"c": [1.5, 2.5], "a": [1, 2]}
    # Corrupt the bytes of `b`'s chunks: they're never read.
    raw = bytearray(path.read_bytes())
    for row_group in range(2):
        meta = pq.ParquetFile(path).metadata.row_group(row_group).column(1)
        start = meta.dictionary_page_offset or meta.data_page_offset
        raw[start : start + meta.total_compressed_size] = (
            b"\xff" * meta.total_compressed_size
        )
    path.write_bytes(bytes(raw))
    assert read_parquet(str(path), ["c", "a"]) == {"c": [1.5, 2.5], "a": [1, 2]}
    with pytest.raises(ValueError, match="may be corrupt"):
        read_parquet(str(path), ["b"])


def _pydict(table: Any) -> dict[str, list[Any]]:
    """`table.to_pydict()`, with maps as lists of `{"key": ..., "value": ...}` structs."""

    def convert(value: Any) -> Any:
        if isinstance(value, list):
            return [convert(v) for v in value]
        if isinstance(value, tuple):
            return {"key": convert(value[0]), "value": convert(value[1])}
        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        return value

    return {k: convert(v) for k, v in table.to_pydict().items()}


def _random_with_nulls(rng: random.Random, values: list[Any]) -> list[Any]:
    return [None if rng.random() < 0.1 else v for v in values]


@pytest.mark.parametrize("data_page_version", ["1.0", "2.0"])
@pytest.mark.parametrize(
    ("dtype", "encoding"),
    [
        (pa.int32(), "DELTA_BINARY_PACKED"),
        (pa.int64(), "DELTA_BINARY_PACKED"),
        (pa.timestamp("us"), "DELTA_BINARY_PACKED"),
        (pa.string(), "DELTA_LENGTH_BYTE_ARRAY"),
        (pa.string(), "DELTA_BYTE_ARRAY"),
        (pa.binary(), "DELTA_BYTE_ARRAY"),
        (pa.float32(), "BYTE_STREAM_SPLIT"),
        (pa.float64(), "BYTE_STREAM_SPLIT"),
        (pa.int32(), "BYTE_STREAM_SPLIT"),
        (pa.int64(), "BYTE_STREAM_SPLIT"),
        (pa.binary(3), "BYTE_STREAM_SPLIT"),
    ],
)
def test_encodings(dtype: Any, encoding: str, data_page_version: str) -> None:
    rng = random.Random(0)  # noqa: S311
    n = 2000
    if pa.types.is_timestamp(dtype):
        values = [rng.randint(0, 2**50) for _ in range(n)]
    elif pa.types.is_integer(dtype):
        bits = dtype.bit_width
        lo, hi = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
        # Extremes make the deltas overflow, and small values vary their bit widths.
        choices = [lo, hi, 0]
        values = [
            rng.choice([*choices, rng.randint(lo, hi), rng.randint(-5, 5)])
            for _ in range(n)
        ]
    elif pa.types.is_floating(dtype):
        values = [rng.uniform(-1e6, 1e6) for _ in range(n)]
    elif pa.types.is_fixed_size_binary(dtype):
        values = [rng.randbytes(3) for _ in range(n)]
    else:
        # Shared prefixes exercise `DELTA_BYTE_ARRAY`'s prefix lengths.
        stems = ["", "prefix", "prefix_longer", "日本"]
        values = [rng.choice(stems) + str(rng.randint(0, 99)) for _ in range(n)]
        if dtype == pa.binary():
            values = [v.encode() for v in values]
    table = pa.table({"a": pa.array(_random_with_nulls(rng, values), dtype)})
    buf = _write(
        table,
        use_dictionary=False,
        column_encoding={"a": encoding},
        data_page_version=data_page_version,
        data_page_size=1024,
    )
    assert read_parquet(buf) == table.to_pydict()


def test_float16() -> None:
    values = [1.5, None, float("inf"), -0.0, 65504.0, 2.0**-24]
    result = read_parquet(_write(pa.table({"a": pa.array(values, pa.float16())})))
    assert result == {"a": values}


def test_lz4_framing() -> None:
    # An LZ4 block: the literals "abc", then a match of 6 bytes, 3 bytes back.
    block = b"\x32abc\x03\x00"
    assert _decompress(LZ4, block, 9) == b"abcabcabc"
    hadoop = len(b"abcabcabc").to_bytes(4, "big") + len(block).to_bytes(4, "big") + block
    assert _decompress(LZ4, hadoop * 2, 18) == b"abcabcabc" * 2
    for corrupt in [b"\x32abc\x00\x00", b"\x32abc\x04\x00", b"\x32abc\x03", b"\x50abc"]:
        with pytest.raises(ValueError, match="may be corrupt"):
            _decompress(LZ4, corrupt, 9)


@pytest.mark.parametrize(
    ("codec", "data", "size"),
    [
        (SNAPPY, b"\x05\x10hello", 4),  # declares 5 bytes, but the page has 4
        (SNAPPY, b"\x03\x08ab", 3),  # a literal longer than the data
        (SNAPPY, b"\x06\x08abc\x01\x00", 6),  # a copy from offset 0
        (SNAPPY, b"\x06\x08abc\x01\x09", 6),  # a copy from before the start
        (SNAPPY, b"\x06\x08abc\x0a\x03", 6),  # a 2-byte offset, cut short
        (GZIP, zlib.compress(b"\x00" * 10_000_000), 100),  # a decompression bomb
    ],
)
def test_decompressed_size_checked(codec: int, data: bytes, size: int) -> None:
    with pytest.raises(ValueError, match="may be corrupt"):
        _decompress(codec, data, size)


def _nested_table(n: int = 200) -> Any:
    rng = random.Random(1)  # noqa: S311

    def maybe(value: Any, p: float = 0.15) -> Any:
        return None if rng.random() < p else value

    def ints(k: int) -> list[Any]:
        return [maybe(rng.randint(-9, 9)) for _ in range(k)]

    struct_type = pa.struct(
        [("x", pa.int64()), ("y", pa.string()), ("z", pa.list_(pa.int64()))]
    )
    required_list = pa.field(
        "required", pa.list_(pa.field("item", pa.int64(), nullable=False)), nullable=False
    )
    columns = {
        "list": pa.array(
            [maybe(ints(rng.randint(0, 4))) for _ in range(n)], pa.list_(pa.int64())
        ),
        "nested_list": pa.array(
            [
                maybe(
                    [
                        maybe([maybe(rng.choice("ab")) for _ in range(rng.randint(0, 3))])
                        for _ in range(rng.randint(0, 3))
                    ]
                )
                for _ in range(n)
            ],
            pa.list_(pa.list_(pa.string())),
        ),
        "struct": pa.array(
            [
                maybe(
                    {
                        "x": maybe(rng.randint(0, 9)),
                        "y": maybe("s"),
                        "z": maybe(ints(rng.randint(0, 2))),
                    }
                )
                for _ in range(n)
            ],
            struct_type,
        ),
        "list_of_structs": pa.array(
            [
                maybe(
                    [
                        maybe({"x": maybe(1), "y": "t", "z": []})
                        for _ in range(rng.randint(0, 3))
                    ]
                )
                for _ in range(n)
            ],
            pa.list_(struct_type),
        ),
        "map": pa.array(
            [
                maybe([(f"k{j}", maybe(j)) for j in range(rng.randint(0, 3))])
                for _ in range(n)
            ],
            pa.map_(pa.string(), pa.int64()),
        ),
        "large_list": pa.array(
            [maybe(ints(rng.randint(0, 3))) for _ in range(n)], pa.large_list(pa.int64())
        ),
        "fixed_size_list": pa.array(
            [maybe(ints(2)) for _ in range(n)], pa.list_(pa.int64(), 2)
        ),
    }
    required = [[rng.randint(0, 9) for _ in range(rng.randint(0, 3))] for _ in range(n)]
    return pa.table(columns).append_column(
        required_list, pa.array(required, required_list.type)
    )


@pytest.mark.parametrize("codec", [_codec_param(c) for c in ["none", "snappy"]])
@pytest.mark.parametrize("data_page_version", ["1.0", "2.0"])
@pytest.mark.parametrize("use_dictionary", [True, False])
def test_nested(codec: str, data_page_version: str, use_dictionary: bool) -> None:  # noqa: FBT001
    table = _nested_table()
    buf = _write(
        table,
        compression=codec,
        data_page_version=data_page_version,
        use_dictionary=use_dictionary,
        row_group_size=64,
        data_page_size=256,
    ).getvalue()
    assert read_parquet(buf) == _pydict(table)
    assert read_parquet(buf, ["map", "list"]) == {
        "map": _pydict(table)["map"],
        "list": _pydict(table)["list"],
    }


def test_nested_written_by_polars() -> None:
    pl = pytest.importorskip("polars")
    df = pl.DataFrame(
        {
            "a": [[1, None], None, []],
            "b": [{"x": 1, "y": [["a"]]}, None, {"x": None, "y": None}],
            "c": [[{"k": "v"}], [], None],
        }
    )
    buf = BytesIO()
    df.write_parquet(buf, compression="snappy")
    buf.seek(0)
    assert read_parquet(buf) == df.to_dict(as_series=False)


def test_legacy_lists() -> None:
    # An unannotated repeated field, two-level `LIST`s whose repeated node is the
    # element (a primitive, a group named `array`, and a group whose only field is
    # repeated), and a three-level one: see the parquet-format spec's
    # backward-compatibility rules for lists.
    schema = _Schema(
        [
            {4: b"root", 5: 5},
            {1: INT32, 3: REPEATED, 4: b"unannotated"},
            {3: OPTIONAL, 4: b"two_level", 5: 1, 6: 3},
            {1: INT32, 3: REPEATED, 4: b"element"},
            {3: OPTIONAL, 4: b"array_struct", 5: 1, 6: 3},
            {3: REPEATED, 4: b"array", 5: 1},
            {1: INT32, 4: b"x"},
            {3: OPTIONAL, 4: b"three_level", 5: 1, 6: 3},
            {3: REPEATED, 4: b"list", 5: 1},
            {1: INT32, 3: OPTIONAL, 4: b"element"},
            {3: OPTIONAL, 4: b"repeated_child", 5: 1, 6: 3},
            {3: REPEATED, 4: b"entries", 5: 1},
            {1: INT32, 3: REPEATED, 4: b"values"},
        ]
    )
    shapes = dict(schema.fields)
    assert all(isinstance(shape, _ListShape) for shape in shapes.values())
    assert [(c.name, c.max_def, c.max_rep) for c in schema.leaves] == [
        ("unannotated", 1, 1),
        ("two_level.element", 2, 1),
        ("array_struct.array.x", 2, 1),
        ("three_level.list.element", 3, 1),
        ("repeated_child.entries.values", 3, 2),
    ]
    # Rows `[1, 2]`, `[]` and `[3]` of the unannotated list.
    leaf = _LeafData([1, 1, 0, 1], [0, 1, 0, 0], [1, 2, 3], 1)
    assert _column_values(shapes["unannotated"], {0: leaf}, 3) == [[1, 2], [], [3]]
    with pytest.raises(ValueError, match="may be corrupt"):  # a row too many
        _column_values(shapes["unannotated"], {0: leaf}, 4)
    leaf = _LeafData([2, 0, 1], [0, 0, 0], [5], 2)
    assert _column_values(shapes["array_struct"], {2: leaf}, 3) == [[{"x": 5}], None, []]
    # A row `[{"values": [1, 2]}, {"values": []}]`, then a null row.
    leaf = _LeafData([3, 3, 2, 0], [0, 2, 1, 0], [1, 2], 3)
    assert _column_values(shapes["repeated_child"], {4: leaf}, 2) == [
        [{"values": [1, 2]}, {"values": []}],
        None,
    ]


@pytest.mark.parametrize(
    "elements",
    [
        [{4: b"root", 5: 1}, {3: OPTIONAL, 4: b"l", 5: 1, 6: 3}, {1: INT32, 4: b"x"}],
        [
            {4: b"root", 5: 1},
            {3: OPTIONAL, 4: b"m", 5: 1, 6: 1},
            {3: REPEATED, 4: b"kv", 5: 3},
            {1: INT32, 4: b"k"},
            {1: INT32, 4: b"v"},
            {1: INT32, 4: b"w"},
        ],
    ],
)
def test_unsupported_nested_layouts(elements: list[dict[int, Any]]) -> None:
    with pytest.raises(NotImplementedError, match="Unsupported layout"):
        _Schema(elements)


def test_page_checksums() -> None:
    raw = bytearray(
        _write(pa.table({"a": [1, 2, 3]}), write_page_checksum=True).getvalue()
    )
    meta = pq.ParquetFile(BytesIO(bytes(raw))).metadata.row_group(0).column(0)
    assert read_parquet(bytes(raw)) == {"a": [1, 2, 3]}
    # The last byte of the column chunk is in its data page's values.
    start = meta.dictionary_page_offset or meta.data_page_offset
    raw[start + meta.total_compressed_size - 1] ^= 1
    with pytest.raises(ValueError, match="checksum mismatch in column 'a'"):
        read_parquet(bytes(raw))


def test_row_counts_checked() -> None:
    buf = _write(pa.table({"a": [1, 2, 3], "b": [[1], [], None]})).getvalue()
    metadata = _buffer_footer(buf)

    def read_chunk(meta: dict[int, Any], column: Any) -> Any:
        return _read_column_chunk(buf, meta, column)

    metadata[3] += 1  # the file's row count
    with pytest.raises(ValueError, match="may be corrupt"):
        _read_parquet(metadata, None, read_chunk)
    metadata[3] -= 1
    metadata[4][0][3] += 1  # and the row group's
    metadata[3] += 1
    for columns in (["a"], ["b"]):
        with pytest.raises(ValueError, match="may be corrupt"):
            _read_parquet(metadata, columns, read_chunk)


def test_invalid_utf8() -> None:
    column = _Column({1: BYTE_ARRAY, 4: b"s", 6: 0})
    with pytest.raises(ValueError, match="'s' holds invalid UTF-8"):
        _decode_plain_values(b"\x01\x00\x00\x00\xff", 0, 1, column)
    column = _Column({1: FIXED_LEN_BYTE_ARRAY, 2: 1, 4: b"s", 6: 0})
    with pytest.raises(ValueError, match="'s' holds invalid UTF-8"):
        _decode_values(b"\xff", 0, PLAIN, 1, column=column, dictionary=None)


def test_columns_deduplicated() -> None:
    assert read_parquet(_write(pa.table({"a": [1]})), ["a", "a"]) == {"a": [1]}


@pytest.mark.parametrize("raw", [b"PAR1", b"PAR1" + b"\xff\xff\x00\x00PAR1"])
def test_truncated_footer(tmp_path: Path, raw: bytes) -> None:
    path = tmp_path / "data.parquet"
    path.write_bytes(raw)
    match = "Not a Parquet file" if len(raw) < 12 else "may be corrupt"
    for source in (raw, str(path)):
        with pytest.raises(ValueError, match=match):
            read_parquet(source)


def test_fuzz() -> None:
    # Whatever the corruption, only `ValueError` or `NotImplementedError` may escape.
    rng = random.Random(0)  # noqa: S311
    table = pa.table(
        {
            "i": [1, None, 3, 4] * 25,
            "s": ["a", "bb", None, "ccc"] * 25,
            "l": [[1, 2], None, [], [None, 5]] * 25,
            "m": pa.array(
                [[("k", 1)], None, [], [("a", None)]] * 25,
                pa.map_(pa.string(), pa.int64()),
            ),
        }
    )
    seeds = [
        _write(
            table, compression=codec, data_page_version=version, row_group_size=40
        ).getvalue()
        for codec in ("none", "snappy", "gzip", "lz4")
        for version in ("1.0", "2.0")
    ]
    seeds.append(
        _write(
            table.select(["i", "s"]),
            use_dictionary=False,
            column_encoding={"i": "DELTA_BINARY_PACKED", "s": "DELTA_BYTE_ARRAY"},
        ).getvalue()
    )
    for _ in range(500):
        raw = bytearray(rng.choice(seeds))
        for _ in range(rng.choice([1, 1, 2, 4])):
            raw[rng.randrange(len(raw))] = rng.randrange(256)
        if rng.random() < 0.1:
            raw = raw[: rng.randrange(len(raw))]
        with suppress(ValueError, NotImplementedError):
            read_parquet(bytes(raw))


# `DELTA_BINARY_PACKED` headers: a block of 128 values in 4 miniblocks, then the value
# count and the first value (zigzag-encoded), then for more values, a block's minimum
# delta and its miniblocks' bit widths (0: no data).
_ONE_ZERO = b"\x80\x01\x04\x01\x00"
_TWO_ZEROS = b"\x80\x01\x04\x02\x00" + b"\x00" + b"\x00" * 4
_ONE_FIVE = b"\x80\x01\x04\x01\x0a"


@pytest.mark.parametrize(
    ("data", "encoding", "physical", "count"),
    [
        (_ONE_ZERO, DELTA_BINARY_PACKED, INT32, 2),  # one value instead of two
        (b"\x80\x01\x03\x01\x00", DELTA_BINARY_PACKED, INT32, 1),  # 128 / 3 miniblocks
        (b"\x80\x01\x20\x01\x00", DELTA_BINARY_PACKED, INT32, 1),  # 4 per miniblock
        (b"\x80\x01\x04\x01\x01", DELTA_LENGTH_BYTE_ARRAY, BYTE_ARRAY, 1),  # length -1
        (_TWO_ZEROS[:-1], DELTA_BINARY_PACKED, INT32, 2),  # a bit width missing
        (
            _TWO_ZEROS[:-4] + b"\x21" + b"\x00" * 3,
            DELTA_BINARY_PACKED,
            INT32,
            2,
        ),  # 33 bits
        (_ONE_ZERO, DELTA_LENGTH_BYTE_ARRAY, BYTE_ARRAY, 2),  # one length instead of two
        (_ONE_FIVE, DELTA_LENGTH_BYTE_ARRAY, BYTE_ARRAY, 1),  # 5 bytes, none there
        (_TWO_ZEROS + _ONE_ZERO, DELTA_BYTE_ARRAY, BYTE_ARRAY, 1),  # two prefixes
        (
            _ONE_FIVE + _ONE_ZERO,
            DELTA_BYTE_ARRAY,
            BYTE_ARRAY,
            1,
        ),  # a 5-byte prefix of b""
    ],
)
def test_corrupt_delta_encodings(
    data: bytes, encoding: int, physical: int, count: int
) -> None:
    column = _Column({1: physical, 4: b"a"})
    with pytest.raises(ValueError, match="may be corrupt"):
        _decode_values(data, 0, encoding, count, column=column, dictionary=None)


def test_corrupt_lz4() -> None:
    block = b"\x32abc\x00\x00"  # a match from offset 0
    hadoop = (9).to_bytes(4, "big") + len(block).to_bytes(4, "big") + block
    wrong_size = (5).to_bytes(4, "big") + (6).to_bytes(4, "big") + b"\x32abc\x03\x00"
    for data in (hadoop, wrong_size, hadoop[:6], (99).to_bytes(4, "big") * 2):
        with pytest.raises(ValueError, match="may be corrupt"):
            _decompress(LZ4, data, 9)


@pytest.mark.parametrize(
    "elements",
    [
        [{4: b"root", 5: 1}, {4: b"s", 5: 0}],  # a struct without fields
        [{4: b"root", 5: 0}, {1: INT32, 4: b"a"}],  # more elements than the tree has
    ],
)
def test_corrupt_schema(elements: list[dict[int, Any]]) -> None:
    with pytest.raises(ValueError, match="may be corrupt"):
        _Schema(elements)


@pytest.mark.parametrize(
    ("leaf", "n_rows"),
    [
        (_LeafData(None, None, [1], 1), 1),  # no levels for a list
        (_LeafData(None, None, [1], 1), 2),  # a row too few
        (_LeafData([1, 1], [0, 2], [1, 2], 1), 1),  # repetition deeper than the schema
    ],
)
def test_corrupt_levels(leaf: _LeafData, n_rows: int) -> None:
    (shape,) = (
        s for _, s in _Schema([{4: b"r", 5: 1}, {1: INT32, 3: REPEATED, 4: b"a"}]).fields
    )
    with pytest.raises(ValueError, match="may be corrupt"):
        _column_values(shape, {0: leaf}, n_rows)


def test_corrupt_v2_level_lengths() -> None:
    column = _Column({1: INT64, 3: OPTIONAL, 4: b"a"})
    header = {1: 3, 2: 3, 4: PLAIN, 5: -1, 6: 0}
    with pytest.raises(ValueError, match="may be corrupt"):
        _read_data_page_v2(b"\x06\x00", header, SNAPPY, 2, column=column, dictionary=None)


def test_corrupt_chunk_count() -> None:
    buf = _write(pa.table({"a": [1], "b": [2]})).getvalue()
    metadata = _buffer_footer(buf)
    metadata[4][0][1].pop()
    with pytest.raises(ValueError, match="may be corrupt"):
        _read_parquet(
            metadata, None, lambda meta, column: _read_column_chunk(buf, meta, column)
        )


def test_float16_of_wrong_width() -> None:
    with pytest.raises(ValueError, match="may be corrupt"):
        _Column({1: FIXED_LEN_BYTE_ARRAY, 2: 3, 4: b"a", 10: {15: {}}})


def test_oversized_delta_count() -> None:
    # 2**40 values in blocks of width-0 miniblocks would take no space at all: the
    # count must be checked against the page's before anything is expanded.
    total, varint = 2**40, bytearray()
    while total >= 0x80:
        varint.append(total & 0x7F | 0x80)
        total >>= 7
    varint.append(total)
    data = b"\x80\x01\x04" + bytes(varint) + b"\x00"
    column = _Column({1: INT64, 4: b"a"})
    with pytest.raises(ValueError, match="may be corrupt"):
        _decode_values(data, 0, DELTA_BINARY_PACKED, 1, column=column, dictionary=None)


def test_level_above_maximum() -> None:
    # An RLE run of one definition level 2, for a column whose maximum is 1: a run's
    # value is a whole byte, unlike bit-packed ones.
    levels = b"\x02\x02"
    column = _Column({1: INT64, 3: OPTIONAL, 4: b"a"})
    header = {1: 1, 2: 0, 4: PLAIN, 5: len(levels), 6: 0, 7: False}
    with pytest.raises(ValueError, match="may be corrupt"):
        _read_data_page_v2(
            levels + b"\x00" * 8, header, SNAPPY, 10, column=column, dictionary=None
        )


def test_variant_group() -> None:
    elements = [
        {4: b"root", 5: 1},
        {3: OPTIONAL, 4: b"v", 5: 2, 10: {16: {}}},
        {1: BYTE_ARRAY, 4: b"metadata"},
        {1: BYTE_ARRAY, 4: b"value"},
    ]
    with pytest.raises(NotImplementedError, match="VARIANT"):
        _Schema(elements)


def test_struct_leaves_disagree_on_nulls() -> None:
    schema = _Schema(
        [
            {4: b"root", 5: 1},
            {3: OPTIONAL, 4: b"s", 5: 2},
            {1: INT32, 3: OPTIONAL, 4: b"a"},
            {1: INT32, 3: OPTIONAL, 4: b"b"},
        ]
    )
    ((_, shape),) = schema.fields
    null = _LeafData([0], None, [], 2)
    assert _column_values(shape, {0: null, 1: null}, 1) == [None]
    with pytest.raises(ValueError, match="may be corrupt"):
        _column_values(shape, {0: null, 1: _LeafData([2], None, [5], 2)}, 1)
