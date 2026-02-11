from __future__ import annotations

from typing import Any

import pytest
from tests.utils import PANDAS_VERSION, Constructor, ConstructorEager, assert_equal_data

import narwhals as nw

NON_NULLABLE_CONSTRUCTOR_NAMES = {
    "pandas_constructor",
    "dask_lazy_p1_constructor",
    "dask_lazy_p2_constructor",
    "modin_constructor",
}


def test_nan(constructor: Constructor) -> None:
    data_na = {"int": [-1, 1, None]}
    df = nw.from_native(constructor(data_na)).with_columns(
        float=nw.col("int").cast(nw.Float64), float_na=nw.col("int") ** 0.5
    )
    result = df.select(
        int=nw.col("int").is_nan(),
        float=nw.col("float").is_nan(),
        float_na=nw.col("float_na").is_nan(),
    )

    expected: dict[str, list[Any]]
    if constructor.__name__ in NON_NULLABLE_CONSTRUCTOR_NAMES:
        # Null values are coerced to NaN for non-nullable datatypes
        expected = {
            "int": [False, False, True],
            "float": [False, False, True],
            "float_na": [True, False, True],
        }
    elif "pandas" in str(constructor) and PANDAS_VERSION >= (3,):
        # NaN values are coerced into NA for nullable datatypes by default
        expected = {
            "int": [False, False, None],
            "float": [False, False, None],
            "float_na": [None, False, None],
        }
    else:
        # Null are preserved and should be differentiated for nullable datatypes
        expected = {
            "int": [False, False, None],
            "float": [False, False, None],
            "float_na": [True, False, None],
        }

    assert_equal_data(result, expected)


def test_nan_series(constructor_eager: ConstructorEager) -> None:
    data_na = {"int": [0, 1, None]}
    df = nw.from_native(constructor_eager(data_na), eager_only=True).with_columns(
        float=nw.col("int").cast(nw.Float64), float_na=nw.col("int") / nw.col("int")
    )

    result = {
        "int": df["int"].is_nan(),
        "float": df["float"].is_nan(),
        "float_na": df["float_na"].is_nan(),
    }
    expected: dict[str, list[Any]]
    if constructor_eager.__name__ in NON_NULLABLE_CONSTRUCTOR_NAMES:
        # Null values are coerced to NaN for non-nullable datatypes
        expected = {
            "int": [False, False, True],
            "float": [False, False, True],
            "float_na": [True, False, True],
        }
    elif "pandas" in str(constructor_eager) and PANDAS_VERSION >= (3,):
        # NaN values are coerced into NA for nullable datatypes by default
        expected = {
            "int": [False, False, None],
            "float": [False, False, None],
            "float_na": [None, False, None],
        }
    else:
        # Null are preserved and should be differentiated for nullable datatypes
        expected = {
            "int": [False, False, None],
            "float": [False, False, None],
            "float_na": [True, False, None],
        }

    assert_equal_data(result, expected)


def test_nan_non_float(constructor: Constructor, request: pytest.FixtureRequest) -> None:
    pytest.importorskip("pyarrow")

    if (
        ("pyspark" in str(constructor))
        or "duckdb" in str(constructor)
        or "ibis" in str(constructor)
    ):
        request.applymarker(pytest.mark.xfail)
    from pyarrow.lib import ArrowNotImplementedError

    from narwhals.exceptions import InvalidOperationError

    data = {"a": ["x", "y"]}
    df = nw.from_native(constructor(data))

    exc = (
        ArrowNotImplementedError
        if "pyarrow_table" in str(constructor)
        else InvalidOperationError
    )

    with pytest.raises(exc):
        df.select(nw.col("a").is_nan()).lazy().collect()


def test_nan_non_float_series(constructor_eager: ConstructorEager) -> None:
    pytest.importorskip("pyarrow")
    from pyarrow.lib import ArrowNotImplementedError

    from narwhals.exceptions import InvalidOperationError

    data = {"a": ["x", "y"]}
    df = nw.from_native(constructor_eager(data), eager_only=True)

    exc = (
        ArrowNotImplementedError
        if "pyarrow_table" in str(constructor_eager)
        else InvalidOperationError
    )

    with pytest.raises(exc):
        df["a"].is_nan()
