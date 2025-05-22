from __future__ import annotations

import narwhals as nw
from tests.utils import Constructor, assert_equal_data

data = {"foo": [1, 2, 3], "BAR": [4, 5, 6]}


def test_to_uppercase(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = df.select((nw.col("foo", "BAR") * 2).name.to_uppercase())
    expected = {"FOO": [2, 4, 6], "BAR": [8, 10, 12]}
    assert_equal_data(result, expected)


def test_to_uppercase_after_alias(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = df.select((nw.col("foo")).alias("alias_for_foo").name.to_uppercase())
    expected = {"FOO": [1, 2, 3]}
    assert_equal_data(result, expected)


def test_to_uppercase_anonymous(constructor: Constructor) -> None:
    df_raw = constructor(data)
    df = nw.from_native(df_raw)
    result = df.select(nw.all().name.to_uppercase())
    expected = {"FOO": [1, 2, 3], "BAR": [4, 5, 6]}
    assert_equal_data(result, expected)
