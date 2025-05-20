from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import Constructor
from tests.utils import assert_equal_data

data = {
    "a": [1, None, 3],
    "b": [None, 1, 2],
}


@pytest.mark.parametrize(
    ("exprs", "expected"),
    [
        ([nw.col("a"), nw.col("b")], {"foo": [None, None, 5]}),
        ([nw.col("a", "b").fill_null(-1)], {"foo": [0, 0, 5]}),
        ([nw.col("a").fill_null(2), nw.col("a", "b").fill_null(-1)], {"foo": [1, 2, 8]}),
    ],
)
def test_reduce_sum(
    constructor: Constructor, exprs: list[nw.Expr], expected: dict[str, list[float]]
) -> None:
    df = nw.from_native(constructor(data))

    result = df.select(nw.reduce(lambda acc, x: acc + x, exprs).alias("foo"))
    assert_equal_data(result, expected)
