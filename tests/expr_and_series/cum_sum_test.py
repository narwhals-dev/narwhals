from __future__ import annotations

import narwhals.stable.v1 as nw
from tests.utils import Constructor
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data

data = {
    "a": [0, 1, 2, None, 4],
    "b": [1, 2, 3, 5, 3],
    "c": [5, 4, 3, 2, 1],
}


def test_cum_sum_simple(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(nw.col("a", "b", "c").cum_sum())
    expected = {
        "a": [0, 1, 3, None, 7],
        "b": [1, 3, 6, 11, 14],
        "c": [5, 9, 12, 14, 15],
    }
    assert_equal_data(result, expected)


def test_cum_sum_simple_series(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    expected = {
        "a": [0, 1, 3, None, 7],
        "b": [1, 3, 6, 11, 14],
        "c": [5, 9, 12, 14, 15],
    }
    result = df.select(
        df["a"].cum_sum(),
        df["b"].cum_sum(),
        df["c"].cum_sum(),
    )
    assert_equal_data(result, expected)
