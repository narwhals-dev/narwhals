from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

import narwhals.stable.v1 as nw_v1
from tests.utils import IBIS_VERSION

pytest.importorskip("polars")
import polars as pl


def test_interchange_schema() -> None:
    df_pl = pl.DataFrame(
        {
            "a": [1, 1, 2],
            "b": [4, 5, 6],
            "c": [4, 5, 6],
            "d": [4, 5, 6],
            "e": [4, 5, 6],
            "f": [4, 5, 6],
            "g": [4, 5, 6],
            "h": [4, 5, 6],
            "i": [4, 5, 6],
            "j": [4, 5, 6],
            "k": ["fdafsd", "fdas", "ad"],
            "l": ["fdafsd", "fdas", "ad"],
            "m": [date(2021, 1, 1), date(2021, 1, 1), date(2021, 1, 1)],
            "n": [True, True, False],
        },
        schema={
            "a": pl.Int64,
            "b": pl.Int32,
            "c": pl.Int16,
            "d": pl.Int8,
            "e": pl.UInt64,
            "f": pl.UInt32,
            "g": pl.UInt16,
            "h": pl.UInt8,
            "i": pl.Float64,
            "j": pl.Float32,
            "k": pl.String,
            "l": pl.Categorical,
            "m": pl.Datetime,
            "n": pl.Boolean,
        },
    )
    df = nw_v1.from_native(df_pl.__dataframe__(), eager_or_interchange_only=True)
    result = df.schema
    expected = {
        "a": nw_v1.Int64,
        "b": nw_v1.Int32,
        "c": nw_v1.Int16,
        "d": nw_v1.Int8,
        "e": nw_v1.UInt64,
        "f": nw_v1.UInt32,
        "g": nw_v1.UInt16,
        "h": nw_v1.UInt8,
        "i": nw_v1.Float64,
        "j": nw_v1.Float32,
        "k": nw_v1.String,
        "l": nw_v1.Categorical,
        "m": nw_v1.Datetime,
        "n": nw_v1.Boolean,
    }
    assert result == expected
    assert df["a"].dtype == nw_v1.Int64


@pytest.mark.filterwarnings("ignore:.*locale specific date formats")
def test_interchange_schema_ibis(
    tmpdir: pytest.TempdirFactory, request: pytest.FixtureRequest
) -> None:  # pragma: no cover
    pytest.importorskip("ibis")
    import ibis

    try:
        ibis.set_backend("duckdb")
    except ImportError:
        request.applymarker(pytest.mark.xfail)
    df_pl = pl.DataFrame(
        {
            "a": [1, 1, 2],
            "b": [4, 5, 6],
            "c": [4, 5, 6],
            "d": [4, 5, 6],
            "e": [4, 5, 6],
            "f": [4, 5, 6],
            "g": [4, 5, 6],
            "h": [4, 5, 6],
            "i": [4, 5, 6],
            "j": [4, 5, 6],
            "k": ["fdafsd", "fdas", "ad"],
            "l": ["fdafsd", "fdas", "ad"],
            "m": [date(2021, 1, 1), date(2021, 1, 1), date(2021, 1, 1)],
            "n": [datetime(2021, 1, 1), datetime(2021, 1, 1), datetime(2021, 1, 1)],
            "o": [True, True, False],
        },
        schema={
            "a": pl.Int64,
            "b": pl.Int32,
            "c": pl.Int16,
            "d": pl.Int8,
            "e": pl.UInt64,
            "f": pl.UInt32,
            "g": pl.UInt16,
            "h": pl.UInt8,
            "i": pl.Float64,
            "j": pl.Float32,
            "k": pl.String,
            "l": pl.Categorical,
            "m": pl.Date,
            "n": pl.Datetime,
            "o": pl.Boolean,
        },
    )
    filepath = str(tmpdir / "file.parquet")  # type: ignore[operator]
    df_pl.write_parquet(filepath)
    tbl = ibis.read_parquet(filepath)
    df = nw_v1.from_native(tbl, eager_or_interchange_only=True)
    result = df.schema
    if IBIS_VERSION > (6, 0, 0):
        expected = {
            "a": nw_v1.Int64,
            "b": nw_v1.Int32,
            "c": nw_v1.Int16,
            "d": nw_v1.Int8,
            "e": nw_v1.UInt64,
            "f": nw_v1.UInt32,
            "g": nw_v1.UInt16,
            "h": nw_v1.UInt8,
            "i": nw_v1.Float64,
            "j": nw_v1.Float32,
            "k": nw_v1.String,
            "l": nw_v1.String,
            "m": nw_v1.Date,
            "n": nw_v1.Datetime,
            "o": nw_v1.Boolean,
        }
    else:
        # Old versions of Ibis would read the file in
        # with different data types
        expected = {
            "a": nw_v1.Int64,
            "b": nw_v1.Int32,
            "c": nw_v1.Int16,
            "d": nw_v1.Int32,
            "e": nw_v1.Int32,
            "f": nw_v1.Int32,
            "g": nw_v1.Int32,
            "h": nw_v1.Int32,
            "i": nw_v1.Float64,
            "j": nw_v1.Float64,
            "k": nw_v1.String,
            "l": nw_v1.String,
            "m": nw_v1.Date,
            "n": nw_v1.Datetime,
            "o": nw_v1.Boolean,
        }
    assert result == expected
    assert df["a"].dtype == nw_v1.Int64
    assert df.columns == list(expected.keys())
    assert df.collect_schema() == expected


def test_interchange_schema_duckdb() -> None:
    pytest.importorskip("duckdb")
    import duckdb

    df_pl = pl.DataFrame(  # noqa: F841
        {
            "a": [1, 1, 2],
            "b": [4, 5, 6],
            "c": [4, 5, 6],
            "d": [4, 5, 6],
            "e": [4, 5, 6],
            "f": [4, 5, 6],
            "g": [4, 5, 6],
            "h": [4, 5, 6],
            "i": [4, 5, 6],
            "j": [4, 5, 6],
            "k": ["fdafsd", "fdas", "ad"],
            "l": ["fdafsd", "fdas", "ad"],
            "m": [date(2021, 1, 1), date(2021, 1, 1), date(2021, 1, 1)],
            "n": [datetime(2021, 1, 1), datetime(2021, 1, 1), datetime(2021, 1, 1)],
            "o": [timedelta(1)] * 3,
            "p": [True, True, False],
        },
        schema={
            "a": pl.Int64,
            "b": pl.Int32,
            "c": pl.Int16,
            "d": pl.Int8,
            "e": pl.UInt64,
            "f": pl.UInt32,
            "g": pl.UInt16,
            "h": pl.UInt8,
            "i": pl.Float64,
            "j": pl.Float32,
            "k": pl.String,
            "l": pl.Categorical,
            "m": pl.Date,
            "n": pl.Datetime,
            "o": pl.Duration,
            "p": pl.Boolean,
        },
    )
    rel = duckdb.sql("select * from df_pl")
    df = nw_v1.from_native(rel, eager_or_interchange_only=True)
    result = df.schema
    expected = {
        "a": nw_v1.Int64,
        "b": nw_v1.Int32,
        "c": nw_v1.Int16,
        "d": nw_v1.Int8,
        "e": nw_v1.UInt64,
        "f": nw_v1.UInt32,
        "g": nw_v1.UInt16,
        "h": nw_v1.UInt8,
        "i": nw_v1.Float64,
        "j": nw_v1.Float32,
        "k": nw_v1.String,
        "l": nw_v1.String,
        "m": nw_v1.Date,
        "n": nw_v1.Datetime,
        "o": nw_v1.Duration,
        "p": nw_v1.Boolean,
    }
    assert result == expected
    assert df["a"].dtype == nw_v1.Int64
    assert df.columns == list(expected.keys())
    assert df.collect_schema() == expected


def test_invalid() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]}).__dataframe__()
    with pytest.raises(
        NotImplementedError, match="is not supported for interchange-level dataframes"
    ):
        nw_v1.from_native(df, eager_or_interchange_only=True).filter([True, False, True])
    with pytest.raises(TypeError, match="Cannot only use `series_only=True`"):
        nw_v1.from_native(df, eager_only=True)
    with pytest.raises(ValueError, match="Invalid parameter combination"):
        nw_v1.from_native(df, eager_only=True, eager_or_interchange_only=True)  # type: ignore[call-overload]
