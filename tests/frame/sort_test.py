from __future__ import annotations

import pytest

import narwhals.stable.v1 as nw
from tests.utils import Constructor
from tests.utils import assert_equal_data


def test_sort(constructor: Constructor) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df = nw.from_native(constructor(data))
    result = df.sort("a", "b")
    expected = {
        "a": [1, 2, 3],
        "b": [4, 6, 4],
        "z": [7.0, 9.0, 8.0],
    }
    assert_equal_data(result, expected)
    result = df.sort("a", "b", descending=[True, False])
    expected = {
        "a": [3, 2, 1],
        "b": [4, 6, 4],
        "z": [8.0, 9.0, 7.0],
    }
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("nulls_last", "expected"),
    [
        (True, {"a": [0, 2, 0, -1], "b": [3, 2, 1, float("nan")]}),
        (False, {"a": [-1, 0, 2, 0], "b": [float("nan"), 3, 2, 1]}),
    ],
)
def test_sort_nulls(
    constructor: Constructor, *, nulls_last: bool, expected: dict[str, float]
) -> None:
    data = {"a": [0, 0, 2, -1], "b": [1, 3, 2, None]}
    df = nw.from_native(constructor(data))
    result = df.sort("b", descending=True, nulls_last=nulls_last)
    assert_equal_data(result, expected)
