from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import pandas as pd
import polars as pl
import pytest

from tests.utils import compare_dicts
from tests.utils import nw

data = {
    "a": [1, 1, 2],
    "b": ["a", "b", "c"],
    "c": [4.0, 5.0, 6.0],
    "d": [True, False, True],
}

if TYPE_CHECKING:
    from narwhals.typing import Expr


@pytest.mark.parametrize("constructor", [pd.DataFrame, pl.DataFrame])
def test_selecctors(constructor: Any) -> None:
    df = nw.from_native(constructor(data))
    result = nw.to_native(df.select(nw.selectors.by_dtype([nw.Int64, nw.Float64]) + 1))
    expected = {"a": [2, 2, 3], "c": [5.0, 6.0, 7.0]}
    compare_dicts(result, expected)


@pytest.mark.parametrize("constructor", [pd.DataFrame, pl.DataFrame])
def test_numeric(constructor: Any) -> None:
    df = nw.from_native(constructor(data))
    result = nw.to_native(df.select(nw.selectors.numeric() + 1))
    expected = {"a": [2, 2, 3], "c": [5.0, 6.0, 7.0]}
    compare_dicts(result, expected)


@pytest.mark.parametrize("constructor", [pd.DataFrame, pl.DataFrame])
def test_boolean(constructor: Any) -> None:
    df = nw.from_native(constructor(data))
    result = nw.to_native(df.select(nw.selectors.boolean()))
    expected = {"d": [True, False, True]}
    compare_dicts(result, expected)


@pytest.mark.parametrize("constructor", [pd.DataFrame, pl.DataFrame])
def test_string(constructor: Any) -> None:
    df = nw.from_native(constructor(data))
    result = nw.to_native(df.select(nw.selectors.string()))
    expected = {"b": ["a", "b", "c"]}
    compare_dicts(result, expected)


def test_categorical() -> None:
    df = nw.from_native(pd.DataFrame(data).astype({"b": "category"}))
    result = nw.to_native(df.select(nw.selectors.categorical()))
    expected = {"b": ["a", "b", "c"]}
    compare_dicts(result, expected)
    df = nw.from_native(pl.DataFrame(data, schema_overrides={"b": pl.Categorical}))
    result = nw.to_native(df.select(nw.selectors.categorical()))
    expected = {"b": ["a", "b", "c"]}
    compare_dicts(result, expected)


@pytest.mark.parametrize("constructor", [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize(
    ("selector", "expected"),
    [
        (nw.selectors.numeric() | nw.selectors.boolean(), ["a", "c", "d"]),
        (nw.selectors.numeric() & nw.selectors.boolean(), []),
        (nw.selectors.numeric() & nw.selectors.by_dtype(nw.Int64), ["a"]),
        (nw.selectors.numeric() | nw.selectors.by_dtype(nw.Int64), ["a", "c"]),
        (~nw.selectors.numeric(), ["b", "d"]),
        (nw.selectors.boolean() & True, ["d"]),
        (nw.selectors.boolean() | True, ["d"]),
        (nw.selectors.numeric() - 1, ["a", "c"]),
        (nw.selectors.all(), ["a", "b", "c", "d"]),
    ],
)
def test_set_ops(constructor: Any, selector: Expr, expected: list[str]) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(selector).columns
    assert sorted(result) == expected


def test_set_ops_invalid() -> None:
    df = nw.from_native(pd.DataFrame(data))
    with pytest.raises(NotImplementedError):
        df.select(1 - nw.selectors.numeric())
    with pytest.raises(NotImplementedError):
        df.select(1 | nw.selectors.numeric())
    with pytest.raises(NotImplementedError):
        df.select(1 & nw.selectors.numeric())
