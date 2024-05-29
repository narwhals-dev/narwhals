from typing import Any

import pandas as pd
import polars as pl
import pytest

import narwhals as nw
from narwhals.selectors import boolean
from narwhals.selectors import by_dtype
from narwhals.selectors import categorical
from narwhals.selectors import numeric
from narwhals.selectors import string
from tests.utils import compare_dicts

data = {
    "a": [1, 1, 2],
    "b": ["a", "b", "c"],
    "c": [4.0, 5.0, 6.0],
    "d": [True, False, True],
}


@pytest.mark.parametrize("constructor", [pd.DataFrame, pl.DataFrame])
def test_selecctors(constructor: Any) -> None:
    df = nw.from_native(constructor(data))
    result = nw.to_native(df.select(by_dtype([nw.Int64, nw.Float64]) + 1))
    expected = {"a": [2, 2, 3], "c": [5.0, 6.0, 7.0]}
    compare_dicts(result, expected)


@pytest.mark.parametrize("constructor", [pd.DataFrame, pl.DataFrame])
def test_numeric(constructor: Any) -> None:
    df = nw.from_native(constructor(data))
    result = nw.to_native(df.select(numeric() + 1))
    expected = {"a": [2, 2, 3], "c": [5.0, 6.0, 7.0]}
    compare_dicts(result, expected)


@pytest.mark.parametrize("constructor", [pd.DataFrame, pl.DataFrame])
def test_boolean(constructor: Any) -> None:
    df = nw.from_native(constructor(data))
    result = nw.to_native(df.select(boolean()))
    expected = {"d": [True, False, True]}
    compare_dicts(result, expected)


@pytest.mark.parametrize("constructor", [pd.DataFrame, pl.DataFrame])
def test_string(constructor: Any) -> None:
    df = nw.from_native(constructor(data))
    result = nw.to_native(df.select(string()))
    expected = {"b": ["a", "b", "c"]}
    compare_dicts(result, expected)


def test_categorical() -> None:
    df = nw.from_native(pd.DataFrame(data).astype({"b": "category"}))
    result = nw.to_native(df.select(categorical()))
    expected = {"b": ["a", "b", "c"]}
    compare_dicts(result, expected)
    df = nw.from_native(pl.DataFrame(data, schema_overrides={"b": pl.Categorical}))
    result = nw.to_native(df.select(categorical()))
    expected = {"b": ["a", "b", "c"]}
    compare_dicts(result, expected)
