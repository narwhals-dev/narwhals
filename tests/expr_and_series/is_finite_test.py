from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import Constructor, ConstructorEager, assert_equal_data

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
