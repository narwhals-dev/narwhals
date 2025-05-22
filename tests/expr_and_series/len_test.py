from __future__ import annotations

import narwhals as nw
from tests.utils import Constructor, ConstructorEager, assert_equal_data


def test_len_no_filter(constructor: Constructor) -> None:
    data = {"a": list("xyz"), "b": [1, 2, None]}
    expected = {"l": [3], "l2": [6], "l3": [3]}
    df = nw.from_native(constructor(data)).select(
        nw.col("a").len().alias("l"),
        (nw.col("a").len() * 2).alias("l2"),
        nw.col("b").len().alias("l3"),
    )

    assert_equal_data(df, expected)


def test_len_chaining(constructor_eager: ConstructorEager) -> None:
    data = {"a": list("xyz"), "b": [1, 2, 1]}
    expected = {"a1": [2], "a2": [1]}
    df = nw.from_native(constructor_eager(data)).select(
        nw.col("a").filter(nw.col("b") == 1).len().alias("a1"),
        nw.col("a").filter(nw.col("b") == 2).len().alias("a2"),
    )

    assert_equal_data(df, expected)


def test_namespace_len(constructor: Constructor) -> None:
    df = nw.from_native(constructor({"a": [1, 2, 3], "b": [4, 5, 6]})).select(
        nw.len(), a=nw.len()
    )
    expected = {"len": [3], "a": [3]}
    assert_equal_data(df, expected)
    df = (
        nw.from_native(constructor({"a": [1, 2, 3], "b": [4, 5, 6]}))
        .filter(nw.col("a") < 0)
        .select(nw.len(), a=nw.len())
    )
    expected = {"len": [0], "a": [0]}
    assert_equal_data(df, expected)


def test_len_series(constructor_eager: ConstructorEager) -> None:
    data = {"a": [1, 2, 1]}
    s = nw.from_native(constructor_eager(data), eager_only=True)["a"]

    assert s.len() == 3
    assert len(s) == 3
