from __future__ import annotations

from typing import Any

import pytest

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts


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
    request: Any, constructor: Any, expr: list[Any], expected: dict[str, list[Any]]
) -> None:
    if "dask" in str(constructor) and int(request.node.callspec.id[-1]) != 1:
        request.applymarker(pytest.mark.xfail)

    data = {"a": [1, 2, 3], "b": [4, 5, 6]}
    df = nw.from_native(constructor(data))
    result = df.select(*expr)
    compare_dicts(result, expected)


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
    request: Any, constructor: Any, expr: list[Any], expected: dict[str, list[Any]]
) -> None:
    if "dask" in str(constructor) and int(request.node.callspec.id[-1]) in [0, 4]:
        request.applymarker(pytest.mark.xfail)

    data = {"a": [1, 2, 3], "b": [4, 5, 6]}
    df = nw.from_native(constructor(data))
    result = df.with_columns(*expr).select(*expected.keys())
    compare_dicts(result, expected)
