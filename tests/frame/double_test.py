from typing import Any

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts


def test_double(constructor: Any) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df = nw.from_native(constructor(data), eager_only=True)
    result = df.with_columns(nw.all() * 2)
    result_native = nw.to_native(result)
    expected = {"a": [2, 6, 4], "b": [8, 8, 12], "z": [14.0, 16.0, 18.0]}
    compare_dicts(result_native, expected)
    result = df.with_columns(nw.col("a").alias("o"), nw.all() * 2)
    result_native = nw.to_native(result)
    expected = {"o": [1, 3, 2], "a": [2, 6, 4], "b": [8, 8, 12], "z": [14.0, 16.0, 18.0]}
    compare_dicts(result_native, expected)
