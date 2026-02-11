from __future__ import annotations

import narwhals as nw
from tests.utils import Constructor, assert_equal_data

data = {"foo": [1, 2, 3], "BAR": [4, 5, 6]}


def test_keep(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = df.select((nw.col("foo", "BAR") * 2).name.keep())
    expected = {k: [e * 2 for e in v] for k, v in data.items()}
    assert_equal_data(result, expected)


def test_keep_after_alias(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = df.select((nw.col("foo")).alias("alias_for_foo").name.keep())
    expected = {"foo": data["foo"]}
    assert_equal_data(result, expected)


def test_keep_anonymous(constructor: Constructor) -> None:
    df_raw = constructor(data)
    df = nw.from_native(df_raw)
    result = df.select("foo").select(nw.all().alias("fdfsad").name.keep())
    expected = {"foo": [1, 2, 3]}
    assert_equal_data(result, expected)
