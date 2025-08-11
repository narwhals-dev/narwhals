from __future__ import annotations

import narwhals as nw
from tests.utils import Constructor, assert_equal_data


def test_top_k(constructor: Constructor) -> None:
    data = {"a": ["a", "f", "a", "d", "b", "c"], "b": [2, 4, 5, 3, 6, 1]}
    df = nw.from_native(constructor(data))
    result = df.top_k(4, by="b")
    expected = {"a": ["b", "a", "f", "d"], "b": [6, 5, 4, 3]}
    assert_equal_data(result, expected)
    df = nw.from_native(constructor(data))
    result = df.top_k(4, by="b", reverse=True)
    expected = {"a": ["c", "a", "d", "f"], "b": [1, 2, 3, 4]}
    assert_equal_data(result, expected)


def test_top_k_by_multiple(constructor: Constructor) -> None:
    data = {
        "a": ["a", "f", "a", "d", "b", "c"],
        "b": [2, 2, 2, 3, 1, 1],
        "sf_c": ["k", "d", "s", "a", "a", "r"],
    }
    df = nw.from_native(constructor(data))
    result = df.top_k(4, by=["b", "sf_c"], reverse=True)
    expected = {
        "a": ["b", "c", "f", "a"],
        "b": [1, 1, 2, 2],
        "sf_c": ["a", "r", "d", "k"],
    }
    assert_equal_data(result, expected)
    data = {
        "a": ["a", "f", "a", "d", "b", "c"],
        "b": [2, 2, 2, 3, 1, 1],
        "sf_c": ["k", "d", "s", "a", "a", "r"],
    }
    df = nw.from_native(constructor(data))
    result = df.top_k(4, by=["b", "sf_c"], reverse=[False, True])
    expected = {
        "a": ["d", "f", "a", "a"],
        "b": [3, 2, 2, 2],
        "sf_c": ["a", "d", "k", "s"],
    }
    assert_equal_data(result, expected)
