from typing import Any

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts


def test_select(constructor: Any) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df = nw.from_native(constructor(data))
    result = df.select("a")
    expected = {"a": [1, 3, 2]}
    compare_dicts(result, expected)


def test_empty_select(constructor: Any) -> None:
    result = nw.from_native(constructor({"a": [1, 2, 3]})).lazy().select()
    assert result.collect().shape == (0, 0)
