from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_series_equal

import narwhals as nw
from narwhals.utils import parse_version

df_pandas = pd.DataFrame({"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]})
if parse_version(pd.__version__) >= parse_version("1.5.0"):
    df_pandas_pyarrow = pd.DataFrame(
        {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    ).astype(
        {
            "a": "Int64[pyarrow]",
            "b": "Int64[pyarrow]",
            "z": "Float64[pyarrow]",
        }
    )
    df_pandas_nullable = pd.DataFrame(
        {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    ).astype(
        {
            "a": "Int64",
            "b": "Int64",
            "z": "Float64",
        }
    )
else:  # pragma: no cover
    # pyarrow/nullable dtypes not supported so far back
    df_pandas_pyarrow = df_pandas
    df_pandas_nullable = df_pandas
df_polars = pl.DataFrame({"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]})
df_lazy = pl.LazyFrame({"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]})


@pytest.mark.parametrize(
    "df_raw", [df_pandas, df_polars, df_pandas_nullable, df_pandas_pyarrow]
)
def test_len(df_raw: Any) -> None:
    result = len(nw.Series(df_raw["a"]))
    assert result == 3
    result = len(nw.LazyFrame(df_raw).collect()["a"])
    assert result == 3


@pytest.mark.parametrize("df_raw", [df_pandas, df_polars])
@pytest.mark.filterwarnings("ignore:np.find_common_type is deprecated:DeprecationWarning")
def test_is_in(df_raw: Any) -> None:
    result = nw.from_native(df_raw["a"], series_only=True).is_in([1, 2])
    assert result[0]
    assert not result[1]
    assert result[2]


@pytest.mark.parametrize("df_raw", [df_pandas, df_polars])
@pytest.mark.filterwarnings("ignore:np.find_common_type is deprecated:DeprecationWarning")
def test_filter(df_raw: Any) -> None:
    result = nw.from_native(df_raw["a"], series_only=True).filter(df_raw["a"] > 1)
    expected = np.array([3, 2])
    assert (result.to_numpy() == expected).all()
    result = nw.DataFrame(df_raw).select(nw.col("a").filter(nw.col("a") > 1))["a"]
    expected = np.array([3, 2])
    assert (result.to_numpy() == expected).all()


@pytest.mark.parametrize("df_raw", [df_pandas, df_polars])
def test_gt(df_raw: Any) -> None:
    s = nw.Series(df_raw["a"])
    result = s > s  # noqa: PLR0124
    assert not result[0]
    assert not result[1]
    assert not result[2]
    result = s > 1
    assert not result[0]
    assert result[1]
    assert result[2]


@pytest.mark.parametrize(
    "df_raw", [df_pandas, df_lazy, df_pandas_nullable, df_pandas_pyarrow]
)
def test_dtype(df_raw: Any) -> None:
    result = nw.LazyFrame(df_raw).collect()["a"].dtype
    assert result == nw.Int64
    assert result.is_numeric()


@pytest.mark.parametrize(
    "df_raw", [df_pandas, df_lazy, df_pandas_nullable, df_pandas_pyarrow]
)
def test_reductions(df_raw: Any) -> None:
    s = nw.LazyFrame(df_raw).collect()["a"]
    assert s.mean() == 2.0
    assert s.std() == 1.0
    assert s.min() == 1
    assert s.max() == 3
    assert s.sum() == 6
    assert nw.to_native(s.is_between(1, 2))[0]
    assert not nw.to_native(s.is_between(1, 2))[1]
    assert nw.to_native(s.is_between(1, 2))[2]
    assert s.n_unique() == 3
    unique = s.unique().sort()
    assert unique[0] == 1
    assert unique[1] == 2
    assert unique[2] == 3
    assert s.alias("foo").name == "foo"


@pytest.mark.parametrize(
    "df_raw", [df_pandas, df_lazy, df_pandas_nullable, df_pandas_pyarrow]
)
def test_boolean_reductions(df_raw: Any) -> None:
    df = nw.LazyFrame(df_raw).select(nw.col("a") > 1)
    assert not df.collect()["a"].all()
    assert df.collect()["a"].any()


@pytest.mark.parametrize("df_raw", [df_pandas, df_lazy])
def test_convert(df_raw: Any) -> None:
    result = nw.LazyFrame(df_raw).collect()["a"].to_numpy()
    assert_array_equal(result, np.array([1, 3, 2]))
    result = nw.LazyFrame(df_raw).collect()["a"].to_pandas()
    assert_series_equal(result, pd.Series([1, 3, 2], name="a"))


def test_dtypes() -> None:
    df = pl.DataFrame(
        {
            "a": [1],
            "b": [1],
            "c": [1],
            "d": [1],
            "e": [1],
            "f": [1],
            "g": [1],
            "h": [1],
            "i": [1],
            "j": [1],
            "k": [1],
            "l": [1],
            "m": [True],
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
            "l": pl.Datetime,
            "m": pl.Boolean,
        },
    )
    result = nw.DataFrame(df).schema
    expected = {
        "a": nw.Int64,
        "b": nw.Int32,
        "c": nw.Int16,
        "d": nw.Int8,
        "e": nw.UInt64,
        "f": nw.UInt32,
        "g": nw.UInt16,
        "h": nw.UInt8,
        "i": nw.Float64,
        "j": nw.Float32,
        "k": nw.String,
        "l": nw.Datetime,
        "m": nw.Boolean,
    }
    assert result == expected
    result_pd = nw.DataFrame(df.to_pandas()).schema
    assert result_pd == expected


def test_cast() -> None:
    df_raw = pl.DataFrame(
        {
            "a": [1],
            "b": [1],
            "c": [1],
            "d": [1],
            "e": [1],
            "f": [1],
            "g": [1],
            "h": [1],
            "i": [1],
            "j": [1],
            "k": [1],
            "l": [1],
            "m": [True],
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
            "l": pl.Datetime,
            "m": pl.Boolean,
        },
    )
    df = nw.DataFrame(df_raw).select(
        nw.col("a").cast(nw.Int32),
        nw.col("b").cast(nw.Int16),
        nw.col("c").cast(nw.Int8),
        nw.col("d").cast(nw.Int64),
        nw.col("e").cast(nw.UInt32),
        nw.col("f").cast(nw.UInt16),
        nw.col("g").cast(nw.UInt8),
        nw.col("h").cast(nw.UInt64),
        nw.col("i").cast(nw.Float32),
        nw.col("j").cast(nw.Float64),
        nw.col("k").cast(nw.String),
        nw.col("l").cast(nw.Datetime),
        nw.col("m").cast(nw.Int8),
        n=nw.col("m").cast(nw.Boolean),
    )
    result = df.schema
    expected = {
        "a": nw.Int32,
        "b": nw.Int16,
        "c": nw.Int8,
        "d": nw.Int64,
        "e": nw.UInt32,
        "f": nw.UInt16,
        "g": nw.UInt8,
        "h": nw.UInt64,
        "i": nw.Float32,
        "j": nw.Float64,
        "k": nw.String,
        "l": nw.Datetime,
        "m": nw.Int8,
        "n": nw.Boolean,
    }
    assert result == expected
    result_pd = nw.DataFrame(df.to_pandas()).schema
    assert result_pd == expected
    result = df.select(
        df["a"].cast(nw.Int32),
        df["b"].cast(nw.Int16),
        df["c"].cast(nw.Int8),
        df["d"].cast(nw.Int64),
        df["e"].cast(nw.UInt32),
        df["f"].cast(nw.UInt16),
        df["g"].cast(nw.UInt8),
        df["h"].cast(nw.UInt64),
        df["i"].cast(nw.Float32),
        df["j"].cast(nw.Float64),
        df["k"].cast(nw.String),
        df["l"].cast(nw.Datetime),
        df["m"].cast(nw.Int8),
        n=df["m"].cast(nw.Boolean),
    ).schema
    expected = {
        "a": nw.Int32,
        "b": nw.Int16,
        "c": nw.Int8,
        "d": nw.Int64,
        "e": nw.UInt32,
        "f": nw.UInt16,
        "g": nw.UInt8,
        "h": nw.UInt64,
        "i": nw.Float32,
        "j": nw.Float64,
        "k": nw.String,
        "l": nw.Datetime,
        "m": nw.Int8,
        "n": nw.Boolean,
    }
    df = nw.from_native(df.to_pandas())  # type: ignore[assignment]
    result_pd = df.select(
        df["a"].cast(nw.Int32),
        df["b"].cast(nw.Int16),
        df["c"].cast(nw.Int8),
        df["d"].cast(nw.Int64),
        df["e"].cast(nw.UInt32),
        df["f"].cast(nw.UInt16),
        df["g"].cast(nw.UInt8),
        df["h"].cast(nw.UInt64),
        df["i"].cast(nw.Float32),
        df["j"].cast(nw.Float64),
        df["k"].cast(nw.String),
        df["l"].cast(nw.Datetime),
        df["m"].cast(nw.Int8),
        n=df["m"].cast(nw.Boolean),
    ).schema
    assert result == expected


def test_to_numpy() -> None:
    s = pd.Series([1, 2, None], dtype="Int64")
    result = nw.Series(s).to_numpy()
    assert result.dtype == "float64"
    result = nw.Series(s).__array__()
    assert result.dtype == "float64"
    assert nw.Series(s).shape == (3,)
