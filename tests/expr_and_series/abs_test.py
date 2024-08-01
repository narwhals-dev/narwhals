from typing import Any

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts


def test_abs(constructor: Any) -> None:
    df = nw.from_native(constructor({"a": [1, 2, 3, -4, 5]}))
    result = df.select(b=nw.col("a").abs())
    expected = {"b": [1, 2, 3, 4, 5]}
    compare_dicts(result, expected)


def test_abs_series(constructor: Any) -> None:
    df = nw.from_native(constructor({"a": [1, 2, 3, -4, 5]}))
    result = df.with_columns(b=nw.col("a").abs()).select("b")
    expected = {"b": [1, 2, 3, 4, 5]}
    compare_dicts(result, expected)
