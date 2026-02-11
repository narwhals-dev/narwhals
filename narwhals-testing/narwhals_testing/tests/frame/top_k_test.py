from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import DUCKDB_VERSION, POLARS_VERSION, Constructor, assert_equal_data


def test_top_k(constructor: Constructor) -> None:
    if "polars" in str(constructor) and POLARS_VERSION < (1, 0):
        # old polars versions do not sort nulls last
        pytest.skip()
    if "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 3):
        pytest.skip()
    data = {"a": ["a", "f", "a", "d", "b", "c"], "b c": [None, None, 2, 3, 6, 1]}
    df = nw.from_native(constructor(data))
    result = df.top_k(4, by="b c")
    expected = {"a": ["a", "b", "c", "d"], "b c": [2, 6, 1, 3]}
    assert_equal_data(result.sort("a"), expected)
    df = nw.from_native(constructor(data))
    result = df.top_k(4, by="b c", reverse=True)
    expected = {"a": ["a", "b", "c", "d"], "b c": [2, 6, 1, 3]}
    assert_equal_data(result.sort(by="a"), expected)


def test_top_k_by_multiple(constructor: Constructor) -> None:
    if "polars" in str(constructor) and POLARS_VERSION < (0, 20, 22):
        # bug in old version
        pytest.skip()
    if "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 3):
        pytest.skip()
    data = {
        "a": ["a", "f", "a", "d", "b", "c"],
        "b": [2, 2, 2, 3, 1, 1],
        "sf_c": ["k", "d", "s", "a", "a", "r"],
    }
    df = nw.from_native(constructor(data))
    result = df.top_k(4, by=["b", "sf_c"], reverse=True)
    expected = {
        "a": ["b", "f", "a", "c"],
        "b": [1, 2, 2, 1],
        "sf_c": ["a", "d", "k", "r"],
    }
    assert_equal_data(result.sort("sf_c"), expected)
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
    assert_equal_data(result.sort("sf_c"), expected)
