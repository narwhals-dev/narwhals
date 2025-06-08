from __future__ import annotations

import narwhals as nw
from tests.utils import Constructor, ConstructorEager, assert_equal_data

data = {
    "i": [0, 1, 2, 3, 4],
    "a": [0, 1, 2, 3, 4],
    "b": [1, 2, 3, 5, 3],
    "c": [5, 4, 3, 2, 1],
}


def test_filter(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data))
    result = df.select(nw.col("a").filter(nw.col("i") < 2, nw.col("c") == 5))
    expected = {"a": [0]}
    assert_equal_data(result, expected)


def test_filter_series(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df.select(df["a"].filter((df["i"] < 2) & (df["c"] == 5)))
    expected = {"a": [0]}
    assert_equal_data(result, expected)
    result_s = df["a"].filter([True, False, False, False, False])
    expected = {"a": [0]}
    assert_equal_data({"a": result_s}, expected)


def test_filter_constraints(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result_added = df.filter(nw.col("i") < 4, b=3)
    expected = {"i": [2], "a": [2], "b": [3], "c": [3]}
    assert_equal_data(result_added, expected)
    result_only = df.filter(i=2, b=3)
    assert_equal_data(result_only, expected)
