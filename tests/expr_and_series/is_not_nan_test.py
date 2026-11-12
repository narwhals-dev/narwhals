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
from tests.utils import PANDAS_VERSION, Constructor, ConstructorEager, assert_equal_data

NON_NULLABLE_CONSTRUCTORS = [
    pandas_constructor,
    dask_lazy_p1_constructor,
    dask_lazy_p2_constructor,
    modin_constructor,
]


def test_not_nan(constructor: Constructor) -> None:
    data_na = {"int": [-1, 1, None]}
    df = nw.from_native(constructor(data_na)).with_columns(
        float=nw.col("int").cast(nw.Float64), float_na=nw.col("int") ** 0.5
    )
    result = df.select(
        int=nw.col("int").is_not_nan(),
        float=nw.col("float").is_not_nan(),
        float_na=nw.col("float_na").is_not_nan(),
    )

    expected: dict[str, list[Any]]
    if any(constructor is c for c in NON_NULLABLE_CONSTRUCTORS):
        # Null values are coerced to NaN for non-nullable datatypes
        expected = {
            "int": [True, True, False],
            "float": [True, True, False],
            "float_na": [False, True, False],
        }
    elif "pandas" in str(constructor) and PANDAS_VERSION >= (3,):
        # NaN values are coerced into NA for nullable datatypes by default
        expected = {
            "int": [True, True, None],
            "float": [True, True, None],
            "float_na": [None, True, None],
        }
    else:
        # Null are preserved and should be differentiated for nullable datatypes
        expected = {
            "int": [True, True, None],
            "float": [True, True, None],
            "float_na": [False, True, None],
        }

    assert_equal_data(result, expected)


def test_not_nan_series(constructor_eager: ConstructorEager) -> None:
    data_na = {"int": [0, 1, None]}
    df = nw.from_native(constructor_eager(data_na), eager_only=True).with_columns(
        float=nw.col("int").cast(nw.Float64), float_na=nw.col("int") / nw.col("int")
    )

    result = {
        "int": df["int"].is_not_nan(),
        "float": df["float"].is_not_nan(),
        "float_na": df["float_na"].is_not_nan(),
    }
    expected: dict[str, list[Any]]
    if any(constructor_eager is c for c in NON_NULLABLE_CONSTRUCTORS):
        # Null values are coerced to NaN for non-nullable datatypes
        expected = {
            "int": [True, True, False],
            "float": [True, True, False],
            "float_na": [False, True, False],
        }
    elif "pandas" in str(constructor_eager) and PANDAS_VERSION >= (3,):
        # NaN values are coerced into NA for nullable datatypes by default
        expected = {
            "int": [True, True, None],
            "float": [True, True, None],
            "float_na": [None, True, None],
        }
    else:
        # Null are preserved and should be differentiated for nullable datatypes
        expected = {
            "int": [True, True, None],
            "float": [True, True, None],
            "float_na": [False, True, None],
        }

    assert_equal_data(result, expected)


def test_not_nan_non_float(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
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
        df.select(nw.col("a").is_not_nan()).lazy().collect()


def test_not_nan_non_float_series(constructor_eager: ConstructorEager) -> None:
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
        df["a"].is_not_nan()
