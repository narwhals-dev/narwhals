from __future__ import annotations

import narwhals as nw
from tests.utils import Constructor, ConstructorEager, assert_equal_data

data = {"a": ["-1", "+1", "1", "12", "123", "99999", "+9999", None]}
expected = {"a": ["-01", "+01", "001", "012", "123", "99999", "+9999", None]}


def test_str_zfill(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(nw.col("a").str.zfill(3))
    assert_equal_data(result, expected)


def test_str_zfill_series(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df.select(df["a"].str.zfill(3))
    assert_equal_data(result, expected)
