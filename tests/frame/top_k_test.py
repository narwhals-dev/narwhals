from __future__ import annotations

import re

import pytest

import narwhals as nw
from tests.utils import (
    DUCKDB_VERSION,
    POLARS_VERSION,
    Constructor,
    assert_equal_data,
    skip_if_no_categorical_ordering,
)


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


@pytest.mark.parametrize("reverse", [[True], [True, False, True]])
def test_top_k_reverse_length_mismatch(
    constructor: Constructor, reverse: list[bool]
) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6]}
    df = nw.from_native(constructor(data))
    with pytest.raises(
        ValueError, match=re.escape("`by` and `reverse` must have the same length.")
    ):
        df.top_k(2, by=["a", "b"], reverse=reverse)


def test_top_k_categorical(constructor: Constructor) -> None:
    # https://github.com/narwhals-dev/narwhals/issues/3841
    skip_if_no_categorical_ordering(constructor)

    data = {"c": ["dog", "cat", "bird"], "n": [1, 2, 3]}
    df = nw.from_native(constructor(data)).cast({"c": nw.Categorical()})
    result = df.top_k(2, by="c").sort("n")
    assert_equal_data(result, {"c": ["dog", "cat"], "n": [1, 2]})
    result = df.top_k(2, by="c", reverse=True).sort("n")
    assert_equal_data(result, {"c": ["cat", "bird"], "n": [2, 3]})
