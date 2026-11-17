from __future__ import annotations

import datetime as dt
import random
import string
import sys
from decimal import Decimal
from io import BytesIO
from typing import TYPE_CHECKING, Any

import pizzlies as pz
import pytest
from pizzlies._plugin.parquet import read_parquet

if TYPE_CHECKING:
    from pathlib import Path

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

CODECS = ["none", "snappy", "gzip", "zstd"]
UTC = dt.timezone.utc


def _codec_param(codec: str) -> Any:
    marks = []
    if codec == "zstd" and sys.version_info < (3, 14):
        marks.append(pytest.mark.skip(reason="stdlib zstd needs Python 3.14+"))
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
    for codec in ("uncompressed", "snappy", "gzip") + (
        ("zstd",) if sys.version_info >= (3, 14) else ()
    ):
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


@pytest.mark.parametrize(
    ("table", "kwargs", "match"),
    [
        ({"a": [[1, 2], [3]]}, {}, "Nested"),
        ({"a": [{"x": 1}]}, {}, "Nested"),
        ({"a": [1, 2]}, {"compression": "brotli"}, "BROTLI"),
        ({"a": [1, 2]}, {"compression": "lz4"}, "LZ4"),
        (
            {"a": [1, 2]},
            {"use_dictionary": False, "column_encoding": "DELTA_BINARY_PACKED"},
            "DELTA_BINARY_PACKED",
        ),
        (
            {"a": [1.0, 2.0]},
            {"use_dictionary": False, "column_encoding": "BYTE_STREAM_SPLIT"},
            "BYTE_STREAM_SPLIT",
        ),
        ({"a": pa.array([1.0], pa.float16())}, {}, "FLOAT16"),
    ],
)
def test_unsupported(table: Any, kwargs: Any, match: str) -> None:
    buf = _write(pa.table(table), **kwargs)
    with pytest.raises(NotImplementedError, match=match):
        read_parquet(buf)
