from __future__ import annotations

from typing import Any

import pytest
from tests.utils import POLARS_VERSION, Constructor, ConstructorEager, assert_equal_data

import narwhals as nw

NON_NULLABLE_CONSTRUCTOR_NAMES = {
    "pandas_constructor",
    "dask_lazy_p1_constructor",
    "dask_lazy_p2_constructor",
    "modin_constructor",
}

data = {"a": [float("nan"), float("inf"), 2.0, None]}


@pytest.mark.filterwarnings("ignore:invalid value encountered in cast")
def test_is_finite_expr(constructor: Constructor) -> None:
    if any(
        x in str(constructor)
        for x in ("polars", "pyarrow_table", "duckdb", "pyspark", "ibis")
    ):
        expected = {"a": [False, False, True, None]}
    elif any(
        x in str(constructor) for x in ("pandas_constructor", "dask", "modin_constructor")
    ):
        expected = {"a": [False, False, True, False]}
    else:  # pandas_nullable_constructor, pandas_pyarrow_constructor, modin_pyarrrow_constructor
        # Here, the 'nan' and None get mangled upon dataframe construction.
        expected = {"a": [None, False, True, None]}

    df = nw.from_native(constructor(data))
    result = df.select(nw.col("a").is_finite())
    assert_equal_data(result, expected)


@pytest.mark.filterwarnings("ignore:invalid value encountered in cast")
def test_is_finite_series(constructor_eager: ConstructorEager) -> None:
    if "polars" in str(constructor_eager) or "pyarrow_table" in str(constructor_eager):
        expected = {"a": [False, False, True, None]}
    elif (
        "pandas_constructor" in str(constructor_eager)
        or "dask" in str(constructor_eager)
        or "modin_constructor" in str(constructor_eager)
    ):
        expected = {"a": [False, False, True, False]}
    else:  # pandas_nullable_constructor, pandas_pyarrow_constructor, modin_pyarrrow_constructor
        expected = {"a": [None, False, True, None]}

    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = {"a": df["a"].is_finite()}

    assert_equal_data(result, expected)


def test_is_finite_integer_column(constructor: Constructor) -> None:
    # Test for https://github.com/narwhals-dev/narwhals/issues/3255
    df = nw.from_native(constructor({"a": [1, 2, 3]}))
    result = df.select(nw.col("a").is_finite())
    assert_equal_data(result, {"a": [True, True, True]})


@pytest.mark.parametrize("data", [[1, 2, None], [1.0, 2.0, None]])
def test_is_finite_column_with_null(constructor: Constructor, data: list[float]) -> None:
    if "polars" in str(constructor) and POLARS_VERSION < (1, 0, 0):
        pytest.skip("need newer polars version")
    df = nw.from_native(constructor({"a": data}))
    result = df.select(nw.col("a").is_finite())

    expected: dict[str, list[Any]]
    if constructor.__name__ in NON_NULLABLE_CONSTRUCTOR_NAMES:
        # Null values are coerced to NaN for non-nullable datatypes
        expected = {"a": [True, True, False]}
    else:
        # Null are preserved and should be differentiated for nullable datatypes
        expected = {"a": [True, True, None]}

    assert_equal_data(result, expected)
