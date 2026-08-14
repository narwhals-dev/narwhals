from __future__ import annotations

import re
from typing import Any

import pytest

import narwhals as nw
from tests.conftest import (
    dask_lazy_p1_constructor,
    dask_lazy_p2_constructor,
    modin_constructor,
    pandas_constructor,
)
from tests.utils import Constructor, ConstructorEager, assert_equal_data

data = {"a": [1, 4, 2, 5]}
data_with_nulls = {"a": [1, 4, 2, None]}

# Boolean columns for these backends can't hold nulls, so a null input gives `False`.
NON_NULLABLE_CONSTRUCTORS = [
    pandas_constructor,
    dask_lazy_p1_constructor,
    dask_lazy_p2_constructor,
    modin_constructor,
]


def test_expr_is_in(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(nw.col("a").is_in([4, 5]))
    expected = {"a": [False, True, False, True]}

    assert_equal_data(result, expected)


def test_expr_is_in_empty_list(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(nw.col("a").is_in([]))
    expected = {"a": [False, False, False, False]}

    assert_equal_data(result, expected)


def test_ser_is_in(constructor_eager: ConstructorEager) -> None:
    ser = nw.from_native(constructor_eager(data), eager_only=True)["a"]
    result = {"a": ser.is_in([4, 5])}
    expected = {"a": [False, True, False, True]}

    assert_equal_data(result, expected)


def test_is_in_other(constructor: Constructor) -> None:
    df_raw = constructor(data)
    msg = re.escape(
        "Narwhals `is_in` doesn't accept expressions as an argument, as opposed to "
        "Polars. You should provide an iterable instead."
    )
    with pytest.raises(NotImplementedError, match=msg):
        nw.from_native(df_raw).with_columns(contains=nw.col("a").is_in("sets"))


def test_filter_is_in_with_series(constructor_eager: ConstructorEager) -> None:
    data = {"a": [1, 4, 2, 5], "b": [1, 0, 2, 0]}
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df.filter(nw.col("a").is_in(df["b"]))
    expected = {"a": [1, 2], "b": [1, 2]}
    assert_equal_data(result, expected)


def test_expr_is_in_nulls(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data_with_nulls))
    result = df.select(nw.col("a").is_in([4, 5]))

    expected: dict[str, list[Any]]
    if any(constructor is c for c in NON_NULLABLE_CONSTRUCTORS):
        expected = {"a": [False, True, False, False]}
    else:
        expected = {"a": [False, True, False, None]}

    assert result.lazy().collect_schema()["a"] == nw.Boolean
    assert_equal_data(result, expected)


def test_expr_is_in_null_in_member_list(constructor: Constructor) -> None:
    # A null member is ignored, so it must not change the mask.
    df = nw.from_native(constructor(data_with_nulls))
    result = df.select(nw.col("a").is_in([4, 5, None]))

    expected: dict[str, list[Any]]
    if any(constructor is c for c in NON_NULLABLE_CONSTRUCTORS):
        expected = {"a": [False, True, False, False]}
    else:
        expected = {"a": [False, True, False, None]}

    assert_equal_data(result, expected)


def test_expr_is_in_empty_list_with_nulls(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data_with_nulls))
    result = df.select(nw.col("a").is_in([]))

    expected: dict[str, list[Any]]
    if any(constructor is c for c in NON_NULLABLE_CONSTRUCTORS):
        expected = {"a": [False, False, False, False]}
    else:
        expected = {"a": [False, False, False, None]}

    assert_equal_data(result, expected)


def test_ser_is_in_nulls(constructor_eager: ConstructorEager) -> None:
    ser = nw.from_native(constructor_eager(data_with_nulls), eager_only=True)["a"]
    result = {"a": ser.is_in([4, 5])}

    expected: dict[str, list[Any]]
    if any(constructor_eager is c for c in NON_NULLABLE_CONSTRUCTORS):
        expected = {"a": [False, True, False, False]}
    else:
        expected = {"a": [False, True, False, None]}

    assert_equal_data(result, expected)


def test_expr_is_in_nulls_dask_nullable() -> None:
    # The `dask` constructor only produces NumPy dtypes, so cover nullable ones here.
    pytest.importorskip("dask")
    import dask.dataframe as dd
    import pandas as pd

    native = dd.from_pandas(pd.DataFrame({"a": pd.array([1, 4, 2, None], dtype="Int64")}))
    result = nw.from_native(native).select(nw.col("a").is_in([4, 5]))

    assert_equal_data(result, {"a": [False, True, False, None]})


def test_filter_is_in_nulls(constructor: Constructor) -> None:
    # `~is_in(...)` keeps the null row on backends which emit `False` instead of null,
    # so the same filter returns a different number of rows depending on the backend.
    df = nw.from_native(constructor(data_with_nulls))
    result = df.filter(~nw.col("a").is_in([4, 5]))

    expected: dict[str, list[Any]]
    if any(constructor is c for c in NON_NULLABLE_CONSTRUCTORS):
        expected = {"a": [1, 2, None]}
    else:
        expected = {"a": [1, 2]}

    assert_equal_data(result, expected)
