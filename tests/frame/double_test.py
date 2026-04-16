from __future__ import annotations

import narwhals as nw
from tests.utils import Constructor, assert_equal_data


def test_double(constructor: Constructor) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8.0, 9.0]}
    df = nw.from_native(constructor(data))

    result = df.with_columns(nw.all() * 2)
    expected = {"a": [2, 6, 4], "b": [8, 8, 12], "z": [14.0, 16.0, 18.0]}
    assert_equal_data(result, expected)

    result = df.with_columns(nw.col("a").alias("o"), nw.all() * 2)
    expected = {"a": [2, 6, 4], "b": [8, 8, 12], "z": [14.0, 16.0, 18.0], "o": [1, 3, 2]}
    assert_equal_data(result, expected)
