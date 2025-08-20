from __future__ import annotations

from contextlib import nullcontext as does_not_raise

import pytest

import narwhals as nw
from narwhals.exceptions import ShapeError
from tests.utils import POLARS_VERSION, Constructor, ConstructorEager, assert_equal_data

data = {"a": [1, 1, 2, 2, 3], "b": [1, 2, 3, 3, 4]}


def test_mode_single_expr_keep_all(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data))
    result = df.select(nw.col("a").mode(keep="all")).sort("a")
    expected = {"a": [1, 2]}
    assert_equal_data(result, expected)


def test_mode_series_keep_all(constructor_eager: ConstructorEager) -> None:
    series = nw.from_native(constructor_eager(data), eager_only=True)["a"]
    result = series.mode(keep="all").sort()
    expected = {"a": [1, 2]}
    assert_equal_data({"a": result}, expected)


def test_mode_different_lengths_keep_all(constructor_eager: ConstructorEager) -> None:
    if "polars" in str(constructor_eager) and POLARS_VERSION < (1, 10):
        pytest.skip()
    df = nw.from_native(constructor_eager(data))
    with pytest.raises(ShapeError):
        df.select(nw.col("a", "b").mode(keep="all"))


def test_mode_expr_keep_any(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(nw.col("a", "b").mode(keep="any"))

    try:
        expected = {"a": [1], "b": [3]}
        assert_equal_data(result, expected)
    except AssertionError:
        expected = {"a": [2], "b": [3]}
        assert_equal_data(result, expected)


def test_mode_expr_keep_all_lazy(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    impl = df.implementation
    not_implemented = {
        nw.Implementation.DUCKDB,
        nw.Implementation.IBIS,
        nw.Implementation.PYSPARK,
        nw.Implementation.PYSPARK_CONNECT,
        nw.Implementation.SQLFRAME,
    }
    msg = "`keep='all'` is not implemented for backend"
    context = (
        pytest.raises(NotImplementedError, match=msg)
        if impl in not_implemented
        else does_not_raise()
    )

    with context:
        result = df.select(nw.col("a").mode(keep="all").sum())
        assert_equal_data(result, {"a": [3]})
