from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import Constructor, assert_equal_data


def test_sort(constructor: Constructor) -> None:
    data = {"an tan": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8.0, 9.0]}
    df = nw.from_native(constructor(data))
    result = df.sort("an tan", "b")
    expected = {"an tan": [1, 2, 3], "b": [4, 6, 4], "z": [7.0, 9.0, 8.0]}
    assert_equal_data(result, expected)
    result = df.sort("an tan", "b", descending=[True, False])
    expected = {"an tan": [3, 2, 1], "b": [4, 6, 4], "z": [8.0, 9.0, 7.0]}
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("nulls_last", "expected"),
    [
        (True, {"antan desc": [0, 2, 0, -1], "b": [3, 2, 1, None]}),
        (False, {"antan desc": [-1, 0, 2, 0], "b": [None, 3, 2, 1]}),
    ],
)
def test_sort_nulls(
    constructor: Constructor, *, nulls_last: bool, expected: dict[str, float]
) -> None:
    data = {"antan desc": [0, 0, 2, -1], "b": [1, 3, 2, None]}
    df = nw.from_native(constructor(data))
    result = df.sort("b", descending=True, nulls_last=nulls_last)
    assert_equal_data(result, expected)
