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


def test_expr_is_in_with_null(constructor: Constructor) -> None:
    # https://github.com/narwhals-dev/narwhals/issues/3851
    data_with_null = {"a": [1, 2, None, 4]}
    df = nw.from_native(constructor(data_with_null))
    result = df.select(nw.col("a").is_in([1, 2]))

    expected: dict[str, list[Any]]
    if any(constructor is c for c in NON_NULLABLE_CONSTRUCTORS):
        # Null values are coerced to NaN for non-nullable datatypes, and
        # `is_in` treats them like a regular (non-matching) value.
        expected = {"a": [True, True, False, False]}
    else:
        expected = {"a": [True, True, None, False]}

    assert_equal_data(result, expected)


def test_filter_is_in_with_null(constructor: Constructor) -> None:
    # https://github.com/narwhals-dev/narwhals/issues/3851
    data_with_null = {"a": [1, 2, None, 4]}
    df = nw.from_native(constructor(data_with_null))
    result = df.filter(~nw.col("a").is_in([1, 2]))

    expected: dict[str, list[Any]]
    if any(constructor is c for c in NON_NULLABLE_CONSTRUCTORS):
        expected = {"a": [None, 4]}
    else:
        expected = {"a": [4]}

    assert_equal_data(result, expected)


def test_expr_is_in_with_null_in_other(constructor: Constructor) -> None:
    # https://github.com/narwhals-dev/narwhals/issues/3851
    # A `None` in `other` shouldn't match anything (not even a null `a`), and
    # shouldn't turn non-null, non-matching values into "unknown" either.
    data_with_null = {"a": [1, 2, None, 4]}
    df = nw.from_native(constructor(data_with_null))
    result = df.select(nw.col("a").is_in([1, None]))

    expected: dict[str, list[Any]]
    if any(constructor is c for c in NON_NULLABLE_CONSTRUCTORS):
        expected = {"a": [True, False, False, False]}
    else:
        expected = {"a": [True, False, None, False]}

    assert_equal_data(result, expected)


def test_expr_is_in_with_only_null_in_other(constructor: Constructor) -> None:
    # https://github.com/narwhals-dev/narwhals/issues/3851
    data_with_null = {"a": [1, 2, None, 4]}
    df = nw.from_native(constructor(data_with_null))
    result = df.select(nw.col("a").is_in([None]))

    expected: dict[str, list[Any]]
    if any(constructor is c for c in NON_NULLABLE_CONSTRUCTORS):
        expected = {"a": [False, False, False, False]}
    else:
        expected = {"a": [False, False, None, False]}

    assert_equal_data(result, expected)
