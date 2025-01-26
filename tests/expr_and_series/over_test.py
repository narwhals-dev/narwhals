from __future__ import annotations

import re
from contextlib import nullcontext as does_not_raise

import pandas as pd
import pytest

import narwhals.stable.v1 as nw
from narwhals.exceptions import LengthChangingExprError
from tests.utils import PANDAS_VERSION
from tests.utils import Constructor
from tests.utils import ConstructorEager
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
    if "duckdb" in str(constructor):
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
    if "duckdb" in str(constructor):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor(data))
    expected = {
        "a": ["a", "a", "b", "b", "b"],
        "b": [1, 2, 3, 3, 5],
        "c": [5, 4, 3, 1, 2],
        "c_min": [5, 4, 1, 1, 2],
    }

    result = df.with_columns(c_min=nw.col("c").min().over("a", "b")).sort("a", "b")
    assert_equal_data(result, expected)


def test_over_cumsum(
    request: pytest.FixtureRequest, constructor_eager: ConstructorEager
) -> None:
    if "pyarrow_table" in str(constructor_eager):
        request.applymarker(pytest.mark.xfail)
    if "pandas_pyarrow" in str(constructor_eager) and PANDAS_VERSION < (2, 1):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor_eager(data_cum))
    expected = {
        "a": ["a", "a", "b", "b", "b"],
        "b": [1, 2, None, 5, 3],
        "c": [5, 4, 3, 2, 1],
        "b_cumsum": [1, 3, None, 5, 8],
        "c_cumsum": [5, 9, 3, 5, 6],
    }

    result = df.with_columns(nw.col("b", "c").cum_sum().over("a").name.suffix("_cumsum"))
    assert_equal_data(result, expected)


def test_over_cumcount(
    request: pytest.FixtureRequest, constructor_eager: ConstructorEager
) -> None:
    if "pyarrow_table" in str(constructor_eager):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor_eager(data_cum))
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


def test_over_cummax(
    request: pytest.FixtureRequest, constructor_eager: ConstructorEager
) -> None:
    if any(x in str(constructor_eager) for x in ("pyarrow_table",)):
        request.applymarker(pytest.mark.xfail)
    if "pandas_pyarrow" in str(constructor_eager) and PANDAS_VERSION < (2, 1):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor_eager(data_cum))
    expected = {
        "a": ["a", "a", "b", "b", "b"],
        "b": [1, 2, None, 5, 3],
        "c": [5, 4, 3, 2, 1],
        "b_cummax": [1, 2, None, 5, 5],
        "c_cummax": [5, 5, 3, 3, 3],
    }
    result = df.with_columns(nw.col("b", "c").cum_max().over("a").name.suffix("_cummax"))
    assert_equal_data(result, expected)


def test_over_cummin(
    request: pytest.FixtureRequest, constructor_eager: ConstructorEager
) -> None:
    if "pyarrow_table" in str(constructor_eager):
        request.applymarker(pytest.mark.xfail)
    if "pandas_pyarrow" in str(constructor_eager) and PANDAS_VERSION < (2, 1):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor_eager(data_cum))
    expected = {
        "a": ["a", "a", "b", "b", "b"],
        "b": [1, 2, None, 5, 3],
        "c": [5, 4, 3, 2, 1],
        "b_cummin": [1, 1, None, 5, 3],
        "c_cummin": [5, 4, 3, 2, 1],
    }

    result = df.with_columns(nw.col("b", "c").cum_min().over("a").name.suffix("_cummin"))
    assert_equal_data(result, expected)


def test_over_cumprod(
    request: pytest.FixtureRequest, constructor_eager: ConstructorEager
) -> None:
    if any(x in str(constructor_eager) for x in ("pyarrow_table",)):
        request.applymarker(pytest.mark.xfail)
    if "pandas_pyarrow" in str(constructor_eager) and PANDAS_VERSION < (2, 1):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor_eager(data_cum))
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


def test_over_anonymous_cumulative(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager({"a": [1, 1, 2], "b": [4, 5, 6]}))
    context = (
        pytest.raises(NotImplementedError)
        if df.implementation.is_pyarrow()
        else pytest.raises(KeyError)  # type: ignore[arg-type]
        if df.implementation.is_modin()
        or (df.implementation.is_pandas() and PANDAS_VERSION < (1, 4))
        # TODO(unassigned): bug in old pandas + modin.
        # df.groupby('a')[['a', 'b']].cum_sum() excludes `'a'` from result
        else does_not_raise()
    )
    with context:
        result = df.with_columns(
            nw.all().cum_sum().over("a").name.suffix("_cum_sum")
        ).sort("a", "b")
        expected = {
            "a": [1, 1, 2],
            "b": [4, 5, 6],
            "a_cum_sum": [1, 2, 2],
            "b_cum_sum": [4, 9, 6],
        }
        assert_equal_data(result, expected)


def test_over_anonymous_reduction(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if "duckdb" in str(constructor) or "pyspark" in str(constructor):
        # TODO(unassigned): we should be able to support these
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor({"a": [1, 1, 2], "b": [4, 5, 6]}))
    context = (
        pytest.raises(NotImplementedError)
        if df.implementation.is_pyarrow()
        or df.implementation.is_pandas_like()
        or df.implementation.is_dask()
        else does_not_raise()
    )
    with context:
        result = (
            nw.from_native(df)
            .with_columns(nw.all().sum().over("a").name.suffix("_sum"))
            .sort("a", "b")
        )
        expected = {
            "a": [1, 1, 2],
            "b": [4, 5, 6],
            "a_sum": [2, 2, 2],
            "b_sum": [9, 9, 6],
        }
        assert_equal_data(result, expected)


def test_over_shift(
    request: pytest.FixtureRequest, constructor_eager: ConstructorEager
) -> None:
    if "pyarrow_table_constructor" in str(constructor_eager):
        request.applymarker(pytest.mark.xfail)
    if ("pyspark" in str(constructor_eager)) or "duckdb" in str(constructor_eager):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor_eager(data))
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


def test_over_raise_len_change(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))

    with pytest.raises(
        LengthChangingExprError,
        match=re.escape("`.over()` can not be used for expressions which change length."),
    ):
        nw.from_native(df).select(nw.col("b").drop_nulls().over("a"))
