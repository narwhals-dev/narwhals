from __future__ import annotations

from typing import Any

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts

data = {"a": ["one", "two", "two"]}


def test_get_categories(constructor: Any) -> None:
    df = nw.from_native(constructor(data), eager_only=True)
    df = df.select(nw.col("a").cast(nw.Categorical))
    expected = {"a": ["one", "two"]}

    result = df.select(nw.col("a").cat.get_categories())
    compare_dicts(result, expected)

    result = df.select(df["a"].cat.get_categories())
    compare_dicts(result, expected)
