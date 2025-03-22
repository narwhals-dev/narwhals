from __future__ import annotations

import pytest

import narwhals.stable.v1 as nw
from tests.utils import Constructor
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data

data = {
    "a": [1, 1, 2, 3, 2],
    "b": [1, 2, 3, 2, 1],
}


def test_is_last_distinct_expr(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data))
    result = df.select(nw.all().is_last_distinct())
    expected = {
        "a": [False, True, False, True, True],
        "b": [False, False, True, True, True],
    }
    assert_equal_data(result, expected)


def test_is_last_distinct_expr_lazy(constructor: Constructor) -> None:
    data = {"a": [1, 1, 2, 3, 2], "b": [1, 2, 3, 2, 1], "i": [0, 1, 2, 3, 4]}
    df = nw.from_native(constructor(data))
    result = (
        df.select(nw.col("a", "b").is_last_distinct().over(_order_by="i"), "i")
        .sort("i")
        .drop("i")
    )
    expected = {
        "a": [False, True, False, True, True],
        "b": [False, False, True, True, True],
    }
    assert_equal_data(result, expected)


def test_is_last_distinct_expr_all(constructor_eager: ConstructorEager) -> None:
    # We can only run this test eagerly, as it relies on row order.
    data = {"a": [1, 1, 2, 3, 2], "b": [1, 2, 3, 2, 1], "i": [0, 1, 2, 3, 4]}
    df = nw.from_native(constructor_eager(data))
    result = df.select(nw.all().is_last_distinct().over(_order_by="i"))
    expected = {
        "a": [False, True, False, True, True],
        "b": [False, False, True, True, True],
        "i": [True, True, True, True, True],
    }
    assert_equal_data(result, expected)


def test_is_last_distinct_expr_lazy_grouped(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if any(x in str(constructor) for x in ("pandas", "pyarrow", "dask")):
        # non-elementary group-by agg
        request.applymarker(pytest.mark.xfail)

    data = {"a": [1, 1, 2, 2, 2], "b": [1, 2, 2, 2, 1], "i": [0, 1, 2, 3, 4]}
    df = nw.from_native(constructor(data))
    result = (
        df.select(nw.col("b").is_last_distinct().over("a", _order_by="i"), "i")
        .sort("i")
        .drop("i")
    )
    expected = {"b": [True, True, False, True, True]}
    assert_equal_data(result, expected)


def test_is_last_distinct_expr_lazy_grouped_nulls(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if any(x in str(constructor) for x in ("pandas", "pyarrow", "dask")):
        # non-elementary group-by agg
        request.applymarker(pytest.mark.xfail)
    data = {"a": [1, 1, 2, 2, 2], "b": [1, 2, 2, 2, 1], "i": [None, 1, 2, 3, 4]}
    df = nw.from_native(constructor(data))
    result = (
        df.select(nw.col("b").is_last_distinct().over("a", _order_by="i"), "i")
        .sort("i")
        .drop("i")
    )
    expected = {"b": [True, True, False, True, True]}
    assert_equal_data(result, expected)


def test_is_last_distinct_series(constructor_eager: ConstructorEager) -> None:
    series = nw.from_native(constructor_eager(data), eager_only=True)["a"]
    result = series.is_last_distinct()
    expected = {
        "a": [False, True, False, True, True],
    }
    assert_equal_data({"a": result}, expected)
