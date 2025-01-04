from __future__ import annotations

import pandas as pd
import pytest

import narwhals.stable.v1 as nw
from tests.utils import PANDAS_VERSION
from tests.utils import Constructor
from tests.utils import assert_equal_data

data = {
    "a": ["a", "a", "b", "b", "b"],
    "b": [1, 2, 3, 5, 3],
    "c": [5, 4, 3, 2, 1],
}

data_cum = {
    "a": ["a", "a", "b", "b", "b"],
    "b": [1, 2, None, 5, 3],
    "c": [5, 4, 3, 2, 1],
}


def test_over_single(request: pytest.FixtureRequest, constructor: Constructor) -> None:
    if "dask_lazy_p2" in str(constructor):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor(data))
    expected = {
        "a": ["a", "a", "b", "b", "b"],
        "b": [1, 2, 3, 5, 3],
        "c": [5, 4, 3, 2, 1],
        "c_max": [5, 5, 3, 3, 3],
    }

    result = df.with_columns(c_max=nw.col("c").max().over("a"))
    assert_equal_data(result, expected)


def test_over_multiple(request: pytest.FixtureRequest, constructor: Constructor) -> None:
    if "dask_lazy_p2" in str(constructor):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor(data))
    expected = {
        "a": ["a", "a", "b", "b", "b"],
        "b": [1, 2, 3, 5, 3],
        "c": [5, 4, 3, 2, 1],
        "c_min": [5, 4, 1, 2, 1],
    }

    result = df.with_columns(c_min=nw.col("c").min().over("a", "b"))
    assert_equal_data(result, expected)


def test_over_invalid(request: pytest.FixtureRequest, constructor: Constructor) -> None:
    if "polars" in str(constructor):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor(data))
    with pytest.raises(ValueError, match="Anonymous expressions"):
        df.with_columns(c_min=nw.all().min().over("a", "b"))


def test_over_cumsum(request: pytest.FixtureRequest, constructor: Constructor) -> None:
    if "pyarrow_table" in str(constructor) or "dask_lazy_p2" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    if "pandas_pyarrow" in str(constructor) and PANDAS_VERSION < (2, 1):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor(data_cum))
    expected = {
        "a": ["a", "a", "b", "b", "b"],
        "b": [1, 2, None, 5, 3],
        "c": [5, 4, 3, 2, 1],
        "b_cumsum": [1, 3, None, 5, 8],
        "c_cumsum": [5, 9, 3, 5, 6],
    }

    result = df.with_columns(nw.col("b", "c").cum_sum().over("a").name.suffix("_cumsum"))
    assert_equal_data(result, expected)


def test_over_cumcount(request: pytest.FixtureRequest, constructor: Constructor) -> None:
    if "pyarrow_table" in str(constructor) or "dask_lazy_p2" in str(constructor):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor(data_cum))
    expected = {
        "a": ["a", "a", "b", "b", "b"],
        "b": [1, 2, None, 5, 3],
        "c": [5, 4, 3, 2, 1],
        "b_cumcount": [1, 2, 0, 1, 2],
        "c_cumcount": [1, 2, 1, 2, 3],
    }

    result = df.with_columns(
        nw.col("b", "c").cum_count().over("a").name.suffix("_cumcount")
    )
    assert_equal_data(result, expected)


def test_over_cummax(request: pytest.FixtureRequest, constructor: Constructor) -> None:
    if any(x in str(constructor) for x in ("pyarrow_table", "dask_lazy_p2", "duckdb")):
        request.applymarker(pytest.mark.xfail)
    if "pandas_pyarrow" in str(constructor) and PANDAS_VERSION < (2, 1):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor(data_cum))
    expected = {
        "a": ["a", "a", "b", "b", "b"],
        "b": [1, 2, None, 5, 3],
        "c": [5, 4, 3, 2, 1],
        "b_cummax": [1, 2, None, 5, 5],
        "c_cummax": [5, 5, 3, 3, 3],
    }
    result = df.with_columns(nw.col("b", "c").cum_max().over("a").name.suffix("_cummax"))
    assert_equal_data(result, expected)


def test_over_cummin(request: pytest.FixtureRequest, constructor: Constructor) -> None:
    if "pyarrow_table" in str(constructor) or "dask_lazy_p2" in str(constructor):
        request.applymarker(pytest.mark.xfail)

    if "pandas_pyarrow" in str(constructor) and PANDAS_VERSION < (2, 1):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor(data_cum))
    expected = {
        "a": ["a", "a", "b", "b", "b"],
        "b": [1, 2, None, 5, 3],
        "c": [5, 4, 3, 2, 1],
        "b_cummin": [1, 1, None, 5, 3],
        "c_cummin": [5, 4, 3, 2, 1],
    }

    result = df.with_columns(nw.col("b", "c").cum_min().over("a").name.suffix("_cummin"))
    assert_equal_data(result, expected)


def test_over_cumprod(request: pytest.FixtureRequest, constructor: Constructor) -> None:
    if any(x in str(constructor) for x in ("pyarrow_table", "dask_lazy_p2", "duckdb")):
        request.applymarker(pytest.mark.xfail)

    if "pandas_pyarrow" in str(constructor) and PANDAS_VERSION < (2, 1):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor(data_cum))
    expected = {
        "a": ["a", "a", "b", "b", "b"],
        "b": [1, 2, None, 5, 3],
        "c": [5, 4, 3, 2, 1],
        "b_cumprod": [1, 2, None, 5, 15],
        "c_cumprod": [5, 20, 3, 6, 6],
    }

    result = df.with_columns(
        nw.col("b", "c").cum_prod().over("a").name.suffix("_cumprod")
    )
    assert_equal_data(result, expected)


def test_over_anonymous() -> None:
    df = pd.DataFrame({"a": [1, 1, 2], "b": [4, 5, 6]})
    with pytest.raises(ValueError, match="Anonymous expressions"):
        nw.from_native(df).select(nw.all().cum_max().over("a"))


def test_over_shift(request: pytest.FixtureRequest, constructor: Constructor) -> None:
    if "pyarrow_table_constructor" in str(
        constructor
    ) or "dask_lazy_p2_constructor" in str(constructor):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor(data))
    expected = {
        "a": ["a", "a", "b", "b", "b"],
        "b": [1, 2, 3, 5, 3],
        "c": [5, 4, 3, 2, 1],
        "b_shift": [None, None, None, None, 3],
    }
    result = df.with_columns(b_shift=nw.col("b").shift(2).over("a"))
    assert_equal_data(result, expected)


def test_over_cum_reverse() -> None:
    df = pd.DataFrame({"a": [1, 1, 2], "b": [4, 5, 6]})

    with pytest.raises(
        NotImplementedError,
        match=r"Cumulative operation with `reverse=True` is not supported",
    ):
        nw.from_native(df).select(nw.col("b").cum_max(reverse=True).over("a"))
