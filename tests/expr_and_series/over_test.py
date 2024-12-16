from __future__ import annotations

from contextlib import nullcontext as does_not_raise

import pyarrow as pa
import pytest

import narwhals.stable.v1 as nw
from tests.utils import Constructor
from tests.utils import assert_equal_data

data = {
    "a": ["a", "a", "b", "b", "b"],
    "b": [1, 2, 3, 5, 3],
    "c": [5, 4, 3, 2, 1],
}


def test_over_single(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    expected = {
        "a": ["a", "a", "b", "b", "b"],
        "b": [1, 2, 3, 5, 3],
        "c": [5, 4, 3, 2, 1],
        "c_max": [5, 5, 3, 3, 3],
    }

    context = (
        pytest.raises(
            NotImplementedError,
            match="`Expr.over` is not supported for Dask backend with multiple partitions.",
        )
        if "dask_lazy_p2" in str(constructor)
        else does_not_raise()
    )

    with context:
        result = df.with_columns(c_max=nw.col("c").max().over("a"))
        assert_equal_data(result, expected)


def test_over_multiple(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    expected = {
        "a": ["a", "a", "b", "b", "b"],
        "b": [1, 2, 3, 5, 3],
        "c": [5, 4, 3, 2, 1],
        "c_min": [5, 4, 1, 2, 1],
    }

    context = (
        pytest.raises(
            NotImplementedError,
            match="`Expr.over` is not supported for Dask backend with multiple partitions.",
        )
        if "dask_lazy_p2" in str(constructor)
        else does_not_raise()
    )

    with context:
        result = df.with_columns(c_min=nw.col("c").min().over("a", "b"))
        assert_equal_data(result, expected)


def test_over_invalid(request: pytest.FixtureRequest, constructor: Constructor) -> None:
    if "polars" in str(constructor):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor(data))
    with pytest.raises(ValueError, match="Anonymous expressions"):
        df.with_columns(c_min=nw.all().min().over("a", "b"))


def test_over_cumsum(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    expected = {
        "a": ["a", "a", "b", "b", "b"],
        "b": [1, 2, 3, 5, 3],
        "c": [5, 4, 3, 2, 1],
        "b_cumsum": [1, 3, 3, 8, 11],
    }

    if "pyarrow_table" in str(constructor):
        context = pytest.raises(
            pa.lib.ArrowKeyError,
            match="No function registered with name: hash_cum_sum",
        )
    elif "dask_lazy_p2" in str(constructor):
        context = pytest.raises(
            NotImplementedError,
            match="`Expr.over` is not supported for Dask backend with multiple partitions.",
        )
    else:
        context = does_not_raise()  # type: ignore[assignment]
    with context:
        result = df.with_columns(b_cumsum=nw.col("b").cum_sum().over("a"))
        assert_equal_data(result, expected)


def test_over_cumcount(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    expected = {
        "a": ["a", "a", "b", "b", "b"],
        "b": [1, 2, 3, 5, 3],
        "c": [5, 4, 3, 2, 1],
        "b_cumcount": [1, 2, 1, 2, 3],
    }

    if "pyarrow_table" in str(constructor):
        context = pytest.raises(
            pa.lib.ArrowKeyError,
            match="No function registered with name: hash_cum_count",
        )
    elif "dask_lazy_p2" in str(constructor):
        context = pytest.raises(
            NotImplementedError,
            match="`Expr.over` is not supported for Dask backend with multiple partitions.",
        )
    else:
        context = does_not_raise()  # type: ignore[assignment]

    with context:
        result = df.with_columns(b_cumcount=nw.col("b").cum_count().over("a"))
        assert_equal_data(result, expected)


def test_over_cumcount_missing_values(constructor: Constructor) -> None:
    data_with_missing_value = {
        "a": ["a", "a", "b", "b", "b"],
        "b": [1, 2, 3, 5, None],
        "c": [5, 4, 3, 2, 1],
    }

    df = nw.from_native(constructor(data_with_missing_value))
    expected = {
        "a": ["a", "a", "b", "b", "b"],
        "b": [1, 2, 3, 5, None],
        "c": [5, 4, 3, 2, 1],
        "b_cumcount": [1, 2, 1, 2, 2],
    }

    if "pyarrow_table" in str(constructor):
        context = pytest.raises(
            pa.lib.ArrowKeyError,
            match="No function registered with name: hash_cum_count",
        )
    elif "dask_lazy_p2" in str(constructor):
        context = pytest.raises(
            NotImplementedError,
            match="`Expr.over` is not supported for Dask backend with multiple partitions.",
        )
    else:
        context = does_not_raise()  # type: ignore[assignment]
    with context:
        result = df.with_columns(b_cumcount=nw.col("b").cum_count().over("a"))
        assert_equal_data(result, expected)


def test_over_cummax(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    expected = {
        "a": ["a", "a", "b", "b", "b"],
        "b": [1, 2, 3, 5, 3],
        "c": [5, 4, 3, 2, 1],
        "b_cummax": [1, 2, 3, 5, 5],
    }

    if "pyarrow_table" in str(constructor):
        context = pytest.raises(
            pa.lib.ArrowKeyError,
            match="No function registered with name: hash_cum_max",
        )
    elif "dask_lazy_p2" in str(constructor):
        context = pytest.raises(
            NotImplementedError,
            match="`Expr.over` is not supported for Dask backend with multiple partitions.",
        )
    else:
        context = does_not_raise()  # type: ignore[assignment]

    with context:
        result = df.with_columns(b_cummax=nw.col("b").cum_max().over("a"))
        assert_equal_data(result, expected)


def test_over_cummin(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    expected = {
        "a": ["a", "a", "b", "b", "b"],
        "b": [1, 2, 3, 5, 3],
        "c": [5, 4, 3, 2, 1],
        "b_cummin": [1, 1, 3, 3, 3],
    }

    if "pyarrow_table" in str(constructor):
        context = pytest.raises(
            pa.lib.ArrowKeyError,
            match="No function registered with name: hash_cum_min",
        )
    elif "dask_lazy_p2" in str(constructor):
        context = pytest.raises(
            NotImplementedError,
            match="`Expr.over` is not supported for Dask backend with multiple partitions.",
        )
    else:
        context = does_not_raise()  # type: ignore[assignment]

    with context:
        result = df.with_columns(b_cummin=nw.col("b").cum_min().over("a"))
        assert_equal_data(result, expected)


def test_over_cumprod(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    expected = {
        "a": ["a", "a", "b", "b", "b"],
        "b": [1, 2, 3, 5, 3],
        "c": [5, 4, 3, 2, 1],
        "b_cumprod": [1, 2, 3, 15, 45],
    }

    if "pyarrow_table" in str(constructor):
        context = pytest.raises(
            pa.lib.ArrowKeyError,
            match="No function registered with name: hash_cum_prod",
        )
    elif "dask_lazy_p2" in str(constructor):
        context = pytest.raises(
            NotImplementedError,
            match="`Expr.over` is not supported for Dask backend with multiple partitions.",
        )
    else:
        context = does_not_raise()  # type: ignore[assignment]

    with context:
        result = df.with_columns(b_cumprod=nw.col("b").cum_prod().over("a"))
        assert_equal_data(result, expected)
