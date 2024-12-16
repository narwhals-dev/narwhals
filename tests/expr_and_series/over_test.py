from __future__ import annotations

import pyarrow as pa
import pytest

import narwhals.stable.v1 as nw
from tests.utils import PY_VERSION
from tests.utils import Constructor
from tests.utils import assert_equal_data

data = {
    "a": ["a", "a", "b", "b", "b"],
    "b": [1, 2, 3, 5, 3],
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
        if (
            "pandas_pyarrow" in str(constructor)
            and PY_VERSION < (3, 10)
            and pa.__version__ < "15.0.0"
        ):
            request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor(data))
    expected = {
        "a": ["a", "a", "b", "b", "b"],
        "b": [1, 2, 3, 5, 3],
        "c": [5, 4, 3, 2, 1],
        "b_cumsum": [1, 3, 3, 8, 11],
    }

    result = df.with_columns(b_cumsum=nw.col("b").cum_sum().over("a"))
    assert_equal_data(result, expected)


def test_over_cumcount(request: pytest.FixtureRequest, constructor: Constructor) -> None:
    if "pyarrow_table" in str(constructor) or "dask_lazy_p2" in str(constructor):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor(data))
    expected = {
        "a": ["a", "a", "b", "b", "b"],
        "b": [1, 2, 3, 5, 3],
        "c": [5, 4, 3, 2, 1],
        "b_cumcount": [1, 2, 1, 2, 3],
    }

    result = df.with_columns(b_cumcount=nw.col("b").cum_count().over("a"))
    assert_equal_data(result, expected)


def test_over_cumcount_missing_values(
    request: pytest.FixtureRequest, constructor: Constructor
) -> None:
    if "pyarrow_table" in str(constructor) or "dask_lazy_p2" in str(constructor):
        request.applymarker(pytest.mark.xfail)

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

    result = df.with_columns(b_cumcount=nw.col("b").cum_count().over("a"))
    assert_equal_data(result, expected)


def test_over_cummax(request: pytest.FixtureRequest, constructor: Constructor) -> None:
    if "pyarrow_table" in str(constructor) or "dask_lazy_p2" in str(constructor):
        request.applymarker(pytest.mark.xfail)
        if (
            "pandas_pyarrow" in str(constructor)
            and PY_VERSION < (3, 10)
            and pa.__version__ < "15.0.0"
        ):
            request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor(data))
    expected = {
        "a": ["a", "a", "b", "b", "b"],
        "b": [1, 2, 3, 5, 3],
        "c": [5, 4, 3, 2, 1],
        "b_cummax": [1, 2, 3, 5, 5],
    }
    result = df.with_columns(b_cummax=nw.col("b").cum_max().over("a"))
    assert_equal_data(result, expected)


def test_over_cummin(request: pytest.FixtureRequest, constructor: Constructor) -> None:
    if "pyarrow_table" in str(constructor) or "dask_lazy_p2" in str(constructor):
        request.applymarker(pytest.mark.xfail)

        if (
            "pandas_pyarrow" in str(constructor)
            and PY_VERSION < (3, 10)
            and pa.__version__ < "15.0.0"
        ):
            request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor(data))
    expected = {
        "a": ["a", "a", "b", "b", "b"],
        "b": [1, 2, 3, 5, 3],
        "c": [5, 4, 3, 2, 1],
        "b_cummin": [1, 1, 3, 3, 3],
    }

    result = df.with_columns(b_cummin=nw.col("b").cum_min().over("a"))
    assert_equal_data(result, expected)


def test_over_cumprod(request: pytest.FixtureRequest, constructor: Constructor) -> None:
    if "pyarrow_table" in str(constructor) or "dask_lazy_p2" in str(constructor):
        request.applymarker(pytest.mark.xfail)

        if (
            "pandas_pyarrow" in str(constructor)
            and PY_VERSION < (3, 10)
            and pa.__version__ < "15.0.0"
        ):
            request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor(data))
    expected = {
        "a": ["a", "a", "b", "b", "b"],
        "b": [1, 2, 3, 5, 3],
        "c": [5, 4, 3, 2, 1],
        "b_cumprod": [1, 2, 3, 15, 45],
    }

    result = df.with_columns(b_cumprod=nw.col("b").cum_prod().over("a"))
    assert_equal_data(result, expected)
