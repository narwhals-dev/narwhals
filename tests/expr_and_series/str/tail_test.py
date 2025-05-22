from __future__ import annotations

import narwhals as nw
from tests.utils import Constructor, ConstructorEager, assert_equal_data

data = {"a": ["foo", "bars"]}


def test_str_tail(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    expected = {"a": ["foo", "ars"]}

    result_frame = df.select(nw.col("a").str.tail(3))
    assert_equal_data(result_frame, expected)


def test_str_tail_series(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    expected = {"a": ["foo", "ars"]}

    result_series = df["a"].str.tail(3)
    assert_equal_data({"a": result_series}, expected)
