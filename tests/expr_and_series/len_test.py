from __future__ import annotations

import pytest

import narwhals.stable.v1 as nw
from tests.utils import Constructor
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data


def test_len_no_filter(constructor: Constructor) -> None:
    data = {"a": list("xyz"), "b": [1, 2, 1]}
    expected = {"l": [3], "l2": [6]}
    df = nw.from_native(constructor(data)).select(
        nw.col("a").len().alias("l"),
        (nw.col("a").len() * 2).alias("l2"),
    )

    assert_equal_data(df, expected)


def test_len_chaining(
    constructor_eager: ConstructorEager, request: pytest.FixtureRequest
) -> None:
    data = {"a": list("xyz"), "b": [1, 2, 1]}
    expected = {"a1": [2], "a2": [1]}
    if "dask" in str(constructor_eager):
        request.applymarker(pytest.mark.xfail)
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
        .select()
        .select(nw.len(), a=nw.len())
    )
    expected = {"len": [0], "a": [0]}
    assert_equal_data(df, expected)


def test_len_series(constructor_eager: ConstructorEager) -> None:
    data = {"a": [1, 2, 1]}
    s = nw.from_native(constructor_eager(data), eager_only=True)["a"]

    assert s.len() == 3
    assert len(s) == 3
