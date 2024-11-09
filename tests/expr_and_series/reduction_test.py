from __future__ import annotations

from typing import Any

import pytest

import narwhals.stable.v1 as nw
from tests.utils import Constructor
from tests.utils import assert_equal_data


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (
            [nw.col("a").min().alias("min"), nw.col("a", "b").mean()],
            {"min": [1], "a": [2], "b": [5]},
        ),
        ([(nw.col("a") + nw.col("b").max()).alias("x")], {"x": [7, 8, 9]}),
        ([nw.col("a"), nw.col("b").min()], {"a": [1, 2, 3], "b": [4, 4, 4]}),
        ([nw.col("a").max(), nw.col("b")], {"a": [3, 3, 3], "b": [4, 5, 6]}),
        (
            [nw.col("a"), nw.col("b").min().alias("min")],
            {"a": [1, 2, 3], "min": [4, 4, 4]},
        ),
    ],
    ids=range(5),
)
def test_scalar_reduction_select(
    constructor: Constructor, expr: list[Any], expected: dict[str, list[Any]]
) -> None:
    data = {"a": [1, 2, 3], "b": [4, 5, 6]}
    df = nw.from_native(constructor(data))
    result = df.select(*expr)
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (
            [nw.col("a").min().alias("min"), nw.col("a", "b").mean()],
            {"min": [1, 1, 1], "a": [2, 2, 2], "b": [5, 5, 5]},
        ),
        ([(nw.col("a") + nw.col("b").max()).alias("x")], {"x": [7, 8, 9]}),
        ([nw.col("a"), nw.col("b").min()], {"a": [1, 2, 3], "b": [4, 4, 4]}),
        ([nw.col("a").max(), nw.col("b")], {"a": [3, 3, 3], "b": [4, 5, 6]}),
        (
            [nw.col("a"), nw.col("b").min().alias("min")],
            {"a": [1, 2, 3], "min": [4, 4, 4]},
        ),
    ],
    ids=range(5),
)
def test_scalar_reduction_with_columns(
    constructor: Constructor, expr: list[Any], expected: dict[str, list[Any]]
) -> None:
    data = {"a": [1, 2, 3], "b": [4, 5, 6]}
    df = nw.from_native(constructor(data))
    result = df.with_columns(*expr).select(*expected.keys())
    assert_equal_data(result, expected)
