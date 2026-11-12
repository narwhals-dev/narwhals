from __future__ import annotations

from typing import Any

import pytest

import narwhals as nw
from tests.conftest import (
    dask_lazy_p1_constructor,
    dask_lazy_p2_constructor,
    modin_constructor,
    pandas_constructor,
)
from tests.utils import POLARS_VERSION, Constructor, ConstructorEager, assert_equal_data

NON_NULLABLE_CONSTRUCTORS = [
    pandas_constructor,
    dask_lazy_p1_constructor,
    dask_lazy_p2_constructor,
    modin_constructor,
]

data = {"a": [float("nan"), float("inf"), 2.0, None]}


@pytest.mark.filterwarnings("ignore:invalid value encountered in cast")
def test_is_infinite_expr(constructor: Constructor) -> None:
    if any(
        x in str(constructor)
        for x in ("polars", "pyarrow_table", "duckdb", "pyspark", "ibis")
    ):
        expected = {"a": [False, True, False, None]}
    elif any(
        x in str(constructor) for x in ("pandas_constructor", "dask", "modin_constructor")
    ):
        expected = {"a": [False, True, False, False]}
    else:  # pandas_nullable_constructor, pandas_pyarrow_constructor, modin_pyarrrow_constructor
        # Here, the 'nan' and None get mangled upon dataframe construction.
        expected = {"a": [None, True, False, None]}

    df = nw.from_native(constructor(data))
    result = df.select(nw.col("a").is_infinite())
    assert_equal_data(result, expected)


@pytest.mark.filterwarnings("ignore:invalid value encountered in cast")
def test_is_infinite_series(constructor_eager: ConstructorEager) -> None:
    if "polars" in str(constructor_eager) or "pyarrow_table" in str(constructor_eager):
        expected = {"a": [False, True, False, None]}
    elif (
        "pandas_constructor" in str(constructor_eager)
        or "dask" in str(constructor_eager)
        or "modin_constructor" in str(constructor_eager)
    ):
        expected = {"a": [False, True, False, False]}
    else:  # pandas_nullable_constructor, pandas_pyarrow_constructor, modin_pyarrrow_constructor
        expected = {"a": [None, True, False, None]}

    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = {"a": df["a"].is_infinite()}

    assert_equal_data(result, expected)


def test_is_infinite_integer_column(constructor: Constructor) -> None:
    df = nw.from_native(constructor({"a": [1, 2, 3]}))
    result = df.select(nw.col("a").is_infinite())
    assert_equal_data(result, {"a": [False, False, False]})


@pytest.mark.parametrize("data", [[1, 2, None], [1.0, 2.0, None]])
def test_is_infinite_column_with_null(
    constructor: Constructor, data: list[float]
) -> None:
    if "polars" in str(constructor) and POLARS_VERSION < (1, 0, 0):
        pytest.skip("need newer polars version")
    df = nw.from_native(constructor({"a": data}))
    result = df.select(nw.col("a").is_infinite())

    expected: dict[str, list[Any]]
    if any(constructor is c for c in NON_NULLABLE_CONSTRUCTORS):
        # Null values are coerced to NaN for non-nullable datatypes
        expected = {"a": [False, False, False]}
    else:
        # Null are preserved and should be differentiated for nullable datatypes
        expected = {"a": [False, False, None]}

    assert_equal_data(result, expected)
