from __future__ import annotations

import narwhals as nw
from tests.utils import Constructor, assert_equal_data

data = {"foo": [1, 2, 3], "BAR": [4, 5, 6]}


def map_func(s: str | None) -> str:
    return str(s)[::-1].lower()


def test_map(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = df.select((nw.col("foo", "BAR") * 2).name.map(function=map_func))
    expected = {"oof": [2, 4, 6], "rab": [8, 10, 12]}
    assert_equal_data(result, expected)


def test_map_after_alias(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = df.select((nw.col("foo")).alias("alias_for_foo").name.map(function=map_func))
    expected = {"oof": data["foo"]}
    assert_equal_data(result, expected)


def test_map_anonymous(constructor: Constructor) -> None:
    df_raw = constructor(data)
    df = nw.from_native(df_raw)
    result = df.select(nw.all().name.map(function=map_func))
    expected = {"oof": [1, 2, 3], "rab": [4, 5, 6]}
    assert_equal_data(result, expected)
