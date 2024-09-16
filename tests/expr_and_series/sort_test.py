from typing import Any

import pytest

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts

data = {"a": [0, 0, 2, -1], "b": [1, 3, 2, None]}


@pytest.mark.parametrize(
    ("descending", "nulls_last", "expected"),
    [
        (True, True, {"b": [3, 2, 1, float("nan")]}),
        (True, False, {"b": [float("nan"), 3, 2, 1]}),
        (False, True, {"b": [1, 2, 3, float("nan")]}),
        (False, False, {"b": [float("nan"), 1, 2, 3]}),
    ],
)
def test_sort_single_expr(
    constructor: Any, descending: Any, nulls_last: Any, expected: Any
) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(nw.col("b").sort(descending=descending, nulls_last=nulls_last))
    compare_dicts(result, expected)


@pytest.mark.parametrize(
    ("descending", "nulls_last", "expected"),
    [
        (True, True, {"a": [0, 0, 2, -1], "b": [3, 2, 1, float("nan")]}),
        (True, False, {"a": [0, 0, 2, -1], "b": [float("nan"), 3, 2, 1]}),
        (False, True, {"a": [0, 0, 2, -1], "b": [1, 2, 3, float("nan")]}),
        (False, False, {"a": [0, 0, 2, -1], "b": [float("nan"), 1, 2, 3]}),
    ],
)
def test_sort_multiple_expr(
    constructor: Any, descending: Any, nulls_last: Any, expected: Any, request: Any
) -> None:
    if "dask" in str(constructor):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor(data))
    result = df.select(
        "a",
        nw.col("b").sort(descending=descending, nulls_last=nulls_last),
    )
    compare_dicts(result, expected)


@pytest.mark.parametrize(
    ("descending", "nulls_last", "expected"),
    [
        (True, True, [3, 2, 1, None]),
        (True, False, [None, 3, 2, 1]),
        (False, True, [1, 2, 3, None]),
        (False, False, [None, 1, 2, 3]),
    ],
)
def test_sort_series(
    constructor_eager: Any, descending: Any, nulls_last: Any, expected: Any
) -> None:
    series = nw.from_native(constructor_eager(data), eager_only=True)["b"]
    result = series.sort(descending=descending, nulls_last=nulls_last)
    assert (
        result == nw.from_native(constructor_eager({"a": expected}), eager_only=True)["a"]
    )
