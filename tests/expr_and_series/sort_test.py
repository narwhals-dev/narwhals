from __future__ import annotations

from typing import Any

import pytest

import narwhals as nw
import narwhals.stable.v1 as nw_v1
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data

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
    constructor_eager: ConstructorEager, descending: Any, nulls_last: Any, expected: Any
) -> None:
    df = nw_v1.from_native(constructor_eager(data), eager_only=True)
    result = df.select(
        "a",
        nw_v1.col("b").sort(descending=descending, nulls_last=nulls_last),
    )
    assert_equal_data(result, expected)
    with pytest.deprecated_call(
        match="is deprecated and will be removed in a future version"
    ):
        df.select(
            "a",
            nw.col("b").sort(descending=descending, nulls_last=nulls_last),
        )


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
    constructor_eager: ConstructorEager, descending: Any, nulls_last: Any, expected: Any
) -> None:
    series = nw_v1.from_native(constructor_eager(data), eager_only=True)["b"]
    result = series.sort(descending=descending, nulls_last=nulls_last)
    assert (
        result
        == nw_v1.from_native(constructor_eager({"a": expected}), eager_only=True)["a"]
    )
