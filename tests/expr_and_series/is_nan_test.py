from __future__ import annotations

import pytest

import narwhals.stable.v1 as nw
from narwhals.exceptions import InvalidOperationError
from tests.conftest import dask_lazy_p2_constructor
from tests.conftest import pandas_constructor
from tests.utils import Constructor
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data

NON_NULLABLE_CONSTRUCTORS = [pandas_constructor, dask_lazy_p2_constructor]


def test_nan(constructor: Constructor) -> None:
    data_na = {"a": [0, 1, None]}
    df = nw.from_native(constructor(data_na)).select(nw.col("a") / nw.col("a"))
    result = df.select(nw.col("a").is_nan())
    if any(constructor is c for c in NON_NULLABLE_CONSTRUCTORS):
        # Null values are coerced to NaN for non-nullable datatypes
        expected = {"a": [True, False, True]}
    else:
        # Null are preserved and should be differentiated for nullable datatypes
        expected = {"a": [True, False, None]}  # type: ignore[list-item]

    assert_equal_data(result, expected)


def test_nan_series(constructor_eager: ConstructorEager) -> None:
    data_na = {"a": [0, 1, None]}
    df = nw.from_native(constructor_eager(data_na), eager_only=True).select(
        nw.col("a") / nw.col("a")
    )
    result = {"a": df["a"].is_nan()}
    if any(constructor_eager is c for c in NON_NULLABLE_CONSTRUCTORS):
        # Null values are coerced to NaN for non-nullable datatypes
        expected = {"a": [True, False, True]}
    else:
        # Null are preserved for nullable datatypes
        expected = {"a": [True, False, None]}  # type: ignore[list-item]

    assert_equal_data(result, expected)


def test_nan_non_float() -> None:
    data = {"a": ["0", "1"]}
    pd_df = nw.from_native(pandas_constructor(data))
    with pytest.raises(InvalidOperationError, match="not supported"):
        pd_df.select(nw.col("a").is_nan())

    dd_df = nw.from_native(dask_lazy_p2_constructor(data))
    with pytest.raises(InvalidOperationError, match="not supported"):
        dd_df.select(nw.col("a").is_nan())
