from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import (
    DUCKDB_VERSION,
    POLARS_VERSION,
    Constructor,
    ConstructorEager,
    assert_equal_data,
)

data = {"a": [1, 1, 2, 3, 2], "b": [1, 2, 3, 2, 1]}


def test_is_first_distinct_expr(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data))
    result = df.select(nw.all().is_first_distinct())
    expected = {
        "a": [True, False, True, True, False],
        "b": [True, True, True, False, False],
    }
    assert_equal_data(result, expected)


def test_is_first_distinct_expr_lazy(constructor: Constructor) -> None:
    if "polars" in str(constructor) and POLARS_VERSION < (1, 10):
        pytest.skip()
    if "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 3):
        pytest.skip()
    data = {"a": [1, 1, 2, 3, 2], "b": [1, 2, 3, 2, 1], "i": [None, 1, 2, 3, 4]}
    df = nw.from_native(constructor(data))
    result = (
        df.select(nw.col("a", "b").is_first_distinct().over(order_by="i"), "i")
        .sort("i")
        .drop("i")
    )
    expected = {
        "a": [True, False, True, True, False],
        "b": [True, True, True, False, False],
    }
    assert_equal_data(result, expected)


def test_is_first_distinct_expr_lazy_grouped(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if "polars" in str(constructor) and POLARS_VERSION < (1, 10):
        pytest.skip()
    if "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 3):
        pytest.skip()
    if any(x in str(constructor) for x in ("dask", "pandas", "pyarrow", "cudf", "modin")):
        request.applymarker(pytest.mark.xfail)
    data = {"a": [1, 1, 2, 2, 2], "b": [1, 3, 3, 2, 3], "i": [0, 1, 2, 3, 4]}
    df = nw.from_native(constructor(data))
    result = (
        df.select(nw.col("b").is_first_distinct().over("a", order_by="i"), "i")
        .sort("i")
        .drop("i")
    )
    expected = {"b": [True, True, True, True, False]}
    assert_equal_data(result, expected)


def test_is_first_distinct_series(constructor_eager: ConstructorEager) -> None:
    series = nw.from_native(constructor_eager(data), eager_only=True)["a"]
    result = series.is_first_distinct()
    expected = {"a": [True, False, True, True, False]}
    assert_equal_data({"a": result}, expected)
