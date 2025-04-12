from __future__ import annotations

from typing import Any

import pytest

import narwhals as nw
from tests.utils import POLARS_VERSION
from tests.utils import Constructor
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data


@pytest.mark.parametrize("expr1", ["a", nw.col("a")])
@pytest.mark.parametrize("expr2", ["b", nw.col("b")])
def test_allh(constructor: Constructor, expr1: Any, expr2: Any) -> None:
    data = {
        "a": [False, False, True],
        "b": [False, True, True],
    }
    df = nw.from_native(constructor(data))
    result = df.select(all=nw.all_horizontal(expr1, expr2))

    expected = {"all": [False, False, True]}
    assert_equal_data(result, expected)


def test_allh_series(constructor_eager: ConstructorEager) -> None:
    data = {
        "a": [False, False, True],
        "b": [False, True, True],
    }
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df.select(all=nw.all_horizontal(df["a"], df["b"]))

    expected = {"all": [False, False, True]}
    assert_equal_data(result, expected)


def test_allh_all(constructor: Constructor) -> None:
    data = {
        "a": [False, False, True],
        "b": [False, True, True],
    }
    df = nw.from_native(constructor(data))
    result = df.select(all=nw.all_horizontal(nw.all()))
    expected = {"all": [False, False, True]}
    assert_equal_data(result, expected)
    result = df.select(nw.all_horizontal(nw.all()))
    expected = {"a": [False, False, True]}
    assert_equal_data(result, expected)


def test_allh_nth(constructor: Constructor) -> None:
    if "polars" in str(constructor) and POLARS_VERSION < (1, 0):
        pytest.skip()
    data = {
        "a": [False, False, True],
        "b": [False, True, True],
    }
    df = nw.from_native(constructor(data))
    result = df.select(nw.all_horizontal(nw.nth(0, 1)))
    expected = {"a": [False, False, True]}
    assert_equal_data(result, expected)
    result = df.select(nw.all_horizontal(nw.col("a"), nw.nth(0)))
    expected = {"a": [False, False, True]}
    assert_equal_data(result, expected)


def test_allh_iterator(constructor: Constructor) -> None:
    def iter_eq(items: Any, /) -> Any:
        for column, value in items:
            yield nw.col(column) == value

    data = {"a": [1, 2, 3, 3, 3], "b": ["b", "b", "a", "a", "b"]}
    df = nw.from_native(constructor(data))
    expr_items = [("a", 3), ("b", "b")]
    expected = {"a": [3], "b": ["b"]}

    eager = nw.all_horizontal(list(iter_eq(expr_items)))
    assert_equal_data(df.filter(eager), expected)
    unpacked = nw.all_horizontal(*iter_eq(expr_items))
    assert_equal_data(df.filter(unpacked), expected)
    lazy = nw.all_horizontal(iter_eq(expr_items))

    assert_equal_data(df.filter(lazy), expected)
    assert_equal_data(df.filter(lazy), expected)
    assert_equal_data(df.filter(lazy), expected)


def test_horizontal_expressions_empty(constructor: Constructor) -> None:
    data = {
        "a": [False, False, True],
        "b": [False, True, True],
    }
    df = nw.from_native(constructor(data))
    with pytest.raises(
        ValueError, match=r"At least one expression must be passed.*all_horizontal"
    ):
        df.select(nw.all_horizontal())
    with pytest.raises(
        ValueError, match=r"At least one expression must be passed.*any_horizontal"
    ):
        df.select(nw.any_horizontal())
    with pytest.raises(
        ValueError, match=r"At least one expression must be passed.*mean_horizontal"
    ):
        df.select(nw.mean_horizontal())
    with pytest.raises(
        ValueError, match=r"At least one expression must be passed.*sum_horizontal"
    ):
        df.select(nw.sum_horizontal())

    with pytest.raises(
        ValueError, match=r"At least one expression must be passed.*max_horizontal"
    ):
        df.select(nw.max_horizontal())

    with pytest.raises(
        ValueError, match=r"At least one expression must be passed.*min_horizontal"
    ):
        df.select(nw.min_horizontal())
