from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import Constructor, assert_equal_data

data = {"a": [1, 3, None, None], "b": [4, None, 6, None], "z": [3, 1, None, None]}
expected_values = [4, 3, 6, None]


@pytest.mark.filterwarnings(r"ignore:.*All-NaN slice encountered:RuntimeWarning")
def test_maxh(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(horizontal_max=nw.max_horizontal("a", nw.col("b"), "z"))
    expected = {"horizontal_max": expected_values}
    assert_equal_data(result, expected)


@pytest.mark.filterwarnings(r"ignore:.*All-NaN slice encountered:RuntimeWarning")
def test_maxh_all(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(nw.max_horizontal(nw.all()), c=nw.max_horizontal(nw.all()))
    expected = {"a": expected_values, "c": expected_values}
    assert_equal_data(result, expected)


def test_maxh_duplicate_output_names(constructor: Constructor) -> None:
    # https://github.com/narwhals-dev/narwhals/issues/3934
    df = nw.from_native(constructor(data))
    exprs = [
        nw.when(nw.col("a") > 1).then(nw.lit(1)),
        nw.when(nw.col("b") > 1).then(nw.lit(2)),
        nw.when(nw.col("z") > 1).then(nw.lit(3)),
    ]
    result = df.select(horizontal_max=nw.max_horizontal(*exprs))
    expected = {"horizontal_max": [3, 1, 2, None]}
    assert_equal_data(result, expected)
