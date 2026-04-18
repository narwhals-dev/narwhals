from __future__ import annotations

import narwhals as nw
from tests.utils import Constructor, ConstructorEager, assert_equal_data

data = {"a": ["foo", "bars"]}


def test_str_tail(nw_frame_constructor: Constructor) -> None:
    df = nw.from_native(nw_frame_constructor(data))
    expected = {"a": ["foo", "ars"]}

    result_frame = df.select(nw.col("a").str.tail(3))
    assert_equal_data(result_frame, expected)


def test_str_tail_series(nw_eager_constructor: ConstructorEager) -> None:
    df = nw.from_native(nw_eager_constructor(data), eager_only=True)
    expected = {"a": ["foo", "ars"]}

    result_series = df["a"].str.tail(3)
    assert_equal_data({"a": result_series}, expected)
