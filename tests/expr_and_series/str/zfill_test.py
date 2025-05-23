from __future__ import annotations

import narwhals as nw
from tests.utils import Constructor, assert_equal_data

data = {"a": ["1", "12", "123"]}


def test_str_zfill(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(nw.col("a").str.zfill(3))
    expected = {"a": ["001", "012", "123"]}
    assert_equal_data(result, expected)
