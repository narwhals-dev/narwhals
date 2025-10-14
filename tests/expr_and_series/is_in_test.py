from __future__ import annotations

import re

import pytest

import narwhals as nw
from tests.utils import (
    Constructor,
    ConstructorEager,
    IntoIterable,
    assert_equal_data,
    assert_equal_series,
)

data = {"a": [1, 4, 2, 5]}


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


def test_expr_is_in_iterable(constructor: Constructor, into_iter_4: IntoIterable) -> None:
    df = nw.from_native(constructor(data))
    iterable = into_iter_4((4, 2))
    result = df.select(nw.col("a").is_in(iterable))
    expected = {"a": [False, True, True, False]}
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


def test_ser_is_in_iterable(
    constructor_eager: ConstructorEager, into_iter_16: IntoIterable
) -> None:
    ser = nw.from_native(constructor_eager(data)).get_column("a")
    iterable = into_iter_16((4, 2))
    result = ser.is_in(iterable)
    expected = [False, True, True, False]
    assert_equal_series(result, expected, "a")
