from typing import Any

import pytest

import narwhals.stable.v1 as nw

data = {"a": [0, 0, 2, -1], "b": [1, 3, 2, None]}


@pytest.mark.parametrize(
    ("descending", "nulls_last", "expected"),
    [
        (True, True, {"a": [0, 0, 2, -1], "b": [3, 2, 1, None]}),
        (True, False, {"a": [0, 0, 2, -1], "b": [None, 3, 2, 1]}),
        (False, True, {"a": [0, 0, 2, -1], "b": [1, 2, 3, None]}),
        (False, False, {"a": [0, 0, 2, -1], "b": [None, 1, 2, 3]}),
    ],
)
def test_sort_expr(
    constructor: Any, descending: Any, nulls_last: Any, expected: Any
) -> None:
    df = nw.from_native(constructor(data), eager_only=True)
    result = nw.to_native(
        df.select(
            "a",
            nw.col("b").sort(descending=descending, nulls_last=nulls_last),
        )
    )
    assert result.equals(constructor(expected))


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
    constructor_lazy: Any, descending: Any, nulls_last: Any, expected: Any
) -> None:
    series = nw.from_native(constructor_lazy(data), eager_only=True)["b"]
    result = series.sort(descending=descending, nulls_last=nulls_last)
    assert (
        result == nw.from_native(constructor_lazy({"a": expected}), eager_only=True)["a"]
    )
