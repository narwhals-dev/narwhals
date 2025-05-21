from __future__ import annotations

import narwhals as nw
from tests.utils import Constructor, ConstructorEager, assert_equal_data

data = {"a": ["foo", "bars"]}


def test_str_head(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(nw.col("a").str.head(3))
    expected = {"a": ["foo", "bar"]}
    assert_equal_data(result, expected)


def test_str_head_series(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    expected = {"a": ["foo", "bar"]}
    result = df.select(df["a"].str.head(3))
    assert_equal_data(result, expected)
