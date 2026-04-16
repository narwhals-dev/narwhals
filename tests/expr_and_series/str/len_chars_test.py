from __future__ import annotations

import narwhals as nw
from tests.utils import Constructor, ConstructorEager, assert_equal_data

data = {"a": ["foo", "foobar", "Café", "345", "東京"]}


def test_str_len_chars(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(nw.col("a").str.len_chars())
    expected = {"a": [3, 6, 4, 3, 2]}
    assert_equal_data(result, expected)


def test_str_len_chars_series(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    expected = {"a": [3, 6, 4, 3, 2]}
    result = df.select(df["a"].str.len_chars())
    assert_equal_data(result, expected)
