from __future__ import annotations

import narwhals.stable.v1 as nw
from tests.utils import Constructor
from tests.utils import compare_dicts


def test_double(constructor: Constructor) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df = nw.from_native(constructor(data))
    result = df.with_columns(nw.all() * 2)
    expected = {"a": [2, 6, 4], "b": [8, 8, 12], "z": [14.0, 16.0, 18.0]}
    compare_dicts(result, expected)


def test_double_alias(constructor: Constructor) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df = nw.from_native(constructor(data))
    result = df.with_columns(nw.col("a").alias("o"), nw.all() * 2)
    expected = {
        "o": [1, 3, 2],
        "a": [2, 6, 4],
        "b": [8, 8, 12],
        "z": [14.0, 16.0, 18.0],
    }
    compare_dicts(result, expected)
