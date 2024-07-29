from typing import Any

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts

data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}


def test_sum_all_expr(constructor: Any) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(nw.all().sum())
    expected = {"a": [6], "b": [14], "z": [24.0]}
    compare_dicts(result, expected)


def test_sum_all_namespace(constructor: Any) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(nw.sum("a", "b", "z"))
    expected = {"a": [6], "b": [14], "z": [24.0]}
    compare_dicts(result, expected)
