from typing import Any

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts


def test_double_selected(constructor: Any) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df = nw.from_native(constructor(data), eager_only=True)
    result = df.select(nw.col("a", "b") * 2)
    result_native = nw.to_native(result)
    expected = {"a": [2, 6, 4], "b": [8, 8, 12]}
    compare_dicts(result_native, expected)
    result = df.select("z", nw.col("a", "b") * 2)
    result_native = nw.to_native(result)
    expected = {"z": [7, 8, 9], "a": [2, 6, 4], "b": [8, 8, 12]}
    compare_dicts(result_native, expected)
    result = df.select("a").select(nw.col("a") + nw.all())
    result_native = nw.to_native(result)
    expected = {"a": [2, 6, 4]}
    compare_dicts(result_native, expected)
