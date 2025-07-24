from __future__ import annotations

from contextlib import nullcontext as does_not_raise

import pytest

import narwhals as nw
from narwhals.exceptions import InvalidOperationError
from tests.utils import Constructor, ConstructorEager, assert_equal_data

data = {"a": [1, 1, 2]}
data_str = {"a": ["x", "x", "y"]}


def test_unique_expr(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    context = (
        pytest.raises(InvalidOperationError)
        if isinstance(df, nw.LazyFrame)
        else does_not_raise()
    )
    with context:
        result = df.select(nw.col("a").unique())
        expected = {"a": [1, 2]}
        assert_equal_data(result, expected)


def test_unique_expr_agg(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if any(x in str(constructor) for x in ("duckdb", "pyspark", "ibis")):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor(data))
    result = df.select(nw.col("a").unique().sum())
    expected = {"a": [3]}
    assert_equal_data(result, expected)


def test_unique_illegal_combination(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    with pytest.raises(InvalidOperationError):
        df.select((nw.col("a").unique() + nw.col("b").unique()).sum())
    with pytest.raises(InvalidOperationError):
        df.select(nw.col("a").unique() + nw.col("b"))


def test_unique_series(constructor_eager: ConstructorEager) -> None:
    series = nw.from_native(constructor_eager(data_str), eager_only=True)["a"]
    result = series.unique(maintain_order=True)
    expected = {"a": ["x", "y"]}
    assert_equal_data({"a": result}, expected)
