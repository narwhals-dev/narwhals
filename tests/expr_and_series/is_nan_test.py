from __future__ import annotations

import os
from contextlib import nullcontext as does_not_raise
from typing import Any

import pytest
from polars.exceptions import ComputeError

import narwhals.stable.v1 as nw
from tests.conftest import dask_lazy_p1_constructor
from tests.conftest import dask_lazy_p2_constructor
from tests.conftest import modin_constructor
from tests.conftest import pandas_constructor
from tests.utils import Constructor
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data

NON_NULLABLE_CONSTRUCTORS = [
    pandas_constructor,
    dask_lazy_p1_constructor,
    dask_lazy_p2_constructor,
    modin_constructor,
]


def test_nan(constructor: Constructor, request: pytest.FixtureRequest) -> None:
    if "duckdb" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    data_na = {"int": [0, 1, None]}
    df = nw.from_native(constructor(data_na)).with_columns(
        float=nw.col("int").cast(nw.Float64), float_na=nw.col("int") / nw.col("int")
    )
    result = df.select(
        int=nw.col("int").is_nan(),
        float=nw.col("float").is_nan(),
        float_na=nw.col("float_na").is_nan(),
    )

    expected: dict[str, list[Any]]
    if any(constructor is c for c in NON_NULLABLE_CONSTRUCTORS):
        # Null values are coerced to NaN for non-nullable datatypes
        expected = {
            "int": [False, False, True],
            "float": [False, False, True],
            "float_na": [True, False, True],
        }
    else:
        # Null are preserved and should be differentiated for nullable datatypes
        expected = {
            "int": [False, False, None],
            "float": [False, False, None],
            "float_na": [True, False, None],
        }

    context = (
        pytest.raises(
            ComputeError, match="NAN is not supported in a Non-floating point type column"
        )
        if "polars_lazy" in str(constructor)
        and os.environ.get("NARWHALS_POLARS_GPU", False)
        else does_not_raise()
    )
    with context:
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
    if any(constructor_eager is c for c in NON_NULLABLE_CONSTRUCTORS):
        # Null values are coerced to NaN for non-nullable datatypes
        expected = {
            "int": [False, False, True],
            "float": [False, False, True],
            "float_na": [True, False, True],
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
    if "duckdb" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    from polars.exceptions import InvalidOperationError as PlInvalidOperationError
    from pyarrow.lib import ArrowNotImplementedError

    from narwhals.exceptions import InvalidOperationError as NwInvalidOperationError

    data = {"a": ["x", "y"]}
    df = nw.from_native(constructor(data))

    exc = NwInvalidOperationError
    if "polars" in str(constructor):
        exc = PlInvalidOperationError
    elif "pyarrow_table" in str(constructor):
        exc = ArrowNotImplementedError

    with pytest.raises(exc):
        df.select(nw.col("a").is_nan()).lazy().collect()


def test_nan_non_float_series(constructor_eager: ConstructorEager) -> None:
    from polars.exceptions import InvalidOperationError as PlInvalidOperationError
    from pyarrow.lib import ArrowNotImplementedError

    from narwhals.exceptions import InvalidOperationError as NwInvalidOperationError

    data = {"a": ["x", "y"]}
    df = nw.from_native(constructor_eager(data), eager_only=True)

    exc = NwInvalidOperationError
    if "polars" in str(constructor_eager):
        exc = PlInvalidOperationError
    elif "pyarrow_table" in str(constructor_eager):
        exc = ArrowNotImplementedError

    with pytest.raises(exc):
        df["a"].is_nan()
