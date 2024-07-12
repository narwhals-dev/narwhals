from datetime import date
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from typing import Any

import pandas as pd
import polars as pl
import pytest

import narwhals.stable.v1 as nw
from narwhals.utils import parse_version

data = {
    "a": [datetime(2020, 1, 1)],
    "b": [datetime(2020, 1, 1, tzinfo=timezone.utc)],
}


def test_schema_comparison() -> None:
    assert {"a": nw.String()} != {"a": nw.Int32()}
    assert {"a": nw.Int32()} == {"a": nw.Int32()}


def test_object() -> None:
    df = pd.DataFrame({"a": [1, 2, 3]}).astype(object)
    result = nw.from_native(df).schema
    assert result["a"] == nw.Object


def test_string_disguised_as_object() -> None:
    df = pd.DataFrame({"a": ["foo", "bar"]}).astype(object)
    result = nw.from_native(df).schema
    assert result["a"] == nw.String


def test_actual_object(constructor: Any) -> None:
    class Foo: ...

    data = {"a": [Foo()]}
    df = nw.from_native(constructor(data))
    result = df.schema
    assert result == {"a": nw.Object}


@pytest.mark.skipif(
    parse_version(pd.__version__) < parse_version("2.0.0"), reason="too old"
)
def test_dtypes() -> None:
    df_pl = pl.DataFrame(
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
            "k": ["1"],
            "l": [1],
            "m": [True],
            "n": [date(2020, 1, 1)],
            "o": [datetime(2020, 1, 1)],
            "p": ["a"],
            "q": [timedelta(1)],
            "r": ["a"],
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
            "n": pl.Date,
            "o": pl.Datetime,
            "p": pl.Categorical,
            "q": pl.Duration,
            "r": pl.Enum(["a", "b"]),
        },
    )
    df = nw.from_native(df_pl, eager_only=True)
    result = df.schema
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
        "n": nw.Date,
        "o": nw.Datetime,
        "p": nw.Categorical,
        "q": nw.Duration,
        "r": nw.Enum,
    }
    assert result == expected
    assert {name: df[name].dtype for name in df.columns} == expected

    # pandas/pyarrow only have categorical, not enum
    expected["r"] = nw.Categorical
    df_pd = df_pl.to_pandas(use_pyarrow_extension_array=True)
    df = nw.from_native(df_pd, eager_only=True)
    result_pd = df.schema
    assert result_pd == expected
    assert {name: df[name].dtype for name in df.columns} == expected
    df_pa = df_pl.to_arrow()
    df = nw.from_native(df_pa, eager_only=True)
    result_pa = df.schema
    assert result_pa == expected
    assert {name: df[name].dtype for name in df.columns} == expected


def test_unknown_dtype() -> None:
    df = pd.DataFrame({"a": pd.period_range("2000", periods=3, freq="M")})
    assert nw.from_native(df).schema == {"a": nw.Unknown}


def test_unknown_dtype_polars() -> None:
    df = pl.DataFrame({"a": [[1, 2, 3]]})
    assert nw.from_native(df).schema == {"a": nw.Unknown}


def test_hash() -> None:
    assert nw.Int64() in {nw.Int64, nw.Int32}
