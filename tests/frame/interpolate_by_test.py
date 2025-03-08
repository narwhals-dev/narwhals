from __future__ import annotations

import narwhals.stable.v1 as nw
from tests.utils import Constructor
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data


def test_interpolate_by_sorted(constructor: Constructor) -> None:
    data_sorted = {
        "by": [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144],
        "target": [0, None, 2, 4, 6, None, 16, 26, 42, None, 110, 178, 288],
    }
    df = nw.from_native(constructor(data_sorted))
    result = df.interpolate_by("target", "by")
    expected = {
        "by": [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144],
        "target": [0, 2, 2, 4, 6, 10, 16, 26, 42, 68, 110, 178, 288],
    }
    assert_equal_data(result, expected)


def test_interpolate_by_unsorted(constructor: Constructor) -> None:
    data_unsorted = {
        "by": [5, 8, 0, 1, 1, 2, 3, 13, 21, 34, 55, 89, 144],
        "target": [None, 16, 0, None, 2, 4, 6, 26, 42, None, 110, 178, 288],
    }
    df = nw.from_native(constructor(data_unsorted))
    result = df.interpolate_by("target", "by")
    expected = {
        "by": [5, 8, 0, 1, 1, 2, 3, 13, 21, 34, 55, 89, 144],
        "target": [10, 16, 0, 2, 2, 4, 6, 26, 42, 68, 110, 178, 288],
    }
    assert_equal_data(result, expected)


def test_interpolate_by_sorted_eager(constructor_eager: ConstructorEager) -> None:
    data_sorted = {
        "by": [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144],
        "target": [0, None, 2, 4, 6, None, 16, 26, 42, None, 110, 178, 288],
    }
    df = nw.from_native(constructor_eager(data_sorted), eager_only=True)
    result = df.interpolate_by("target", "by")
    expected = {
        "by": [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144],
        "target": [0, 2, 2, 4, 6, 10, 16, 26, 42, 68, 110, 178, 288],
    }
    assert_equal_data(result, expected)


def test_interpolate_by_unsorted_eager(
    constructor_eager: ConstructorEager,
) -> None:
    data_unsorted = {
        "by": [5, 8, 0, 1, 1, 2, 3, 13, 21, 34, 55, 89, 144],
        "target": [None, 16, 0, None, 2, 4, 6, 26, 42, None, 110, 178, 288],
    }
    df = nw.from_native(constructor_eager(data_unsorted), eager_only=True)
    result = df.interpolate_by("target", "by")
    expected = {
        "by": [5, 8, 0, 1, 1, 2, 3, 13, 21, 34, 55, 89, 144],
        "target": [10, 16, 0, 2, 2, 4, 6, 26, 42, 68, 110, 178, 288],
    }
    assert_equal_data(result, expected)
