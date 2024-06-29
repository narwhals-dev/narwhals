from datetime import datetime
from datetime import timezone
from typing import Any

import pandas as pd
import polars as pl
import pytest

import narwhals as nw

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


@pytest.mark.parametrize("constructor", [pd.DataFrame, pl.DataFrame])
def test_actual_object(constructor: Any) -> None:
    class Foo: ...

    data = {"a": [Foo()]}
    df = nw.from_native(constructor(data))
    result = df.schema
    assert result == {"a": nw.Object}


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
            "k": ["1"],
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
    result_pa = nw.DataFrame(df.to_arrow()).schema
    assert result_pa == expected
