from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw
from tests.utils import Constructor, ConstructorEager, assert_equal_data

if TYPE_CHECKING:
    from collections.abc import Callable

data = {"a": [1, 4, 2, 5]}

# `is_in` documents and accepts any non-str/bytes `Iterable`, but only `list` was
# tested. One-shot iterators (e.g. generators) used to raise on some backends.
iterable_factories: list[Callable[[], Any]] = [
    lambda: (4, 5),
    lambda: {4, 5},
    lambda: range(4, 6),
    lambda: (x for x in (4, 5)),  # one-shot generator (the regression)
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


@pytest.mark.parametrize("make_other", iterable_factories)
def test_expr_is_in_iterables(
    constructor: Constructor, make_other: Callable[[], Any]
) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(nw.col("a").is_in(make_other()))
    expected = {"a": [False, True, False, True]}

    assert_equal_data(result, expected)


@pytest.mark.parametrize("make_other", iterable_factories)
def test_ser_is_in_iterables(
    constructor_eager: ConstructorEager, make_other: Callable[[], Any]
) -> None:
    ser = nw.from_native(constructor_eager(data), eager_only=True)["a"]
    result = {"a": ser.is_in(make_other())}
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
