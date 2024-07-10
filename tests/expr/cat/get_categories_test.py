from __future__ import annotations

from typing import Any

import pyarrow as pa

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts

data = {"a": ["one", "two", "two"]}


def test_get_categories(constructor: Any) -> None:
    df = nw.from_native(constructor(data), eager_only=True)
    df = df.select(nw.col("a").cast(nw.Categorical))
    result = df.select(nw.col("a").cat.get_categories())
    expected = {"a": ["one", "two"]}
    compare_dicts(result, expected)
    result = df.select(df["a"].cat.get_categories())
    compare_dicts(result, expected)


def test_get_categories_pyarrow() -> None:
    # temporary test until we have `cast` in pyarrow - later, fuse
    # this with test above
    table = pa.table(
        {"a": pa.array(["a", "b", None, "d"], pa.dictionary(pa.int64(), pa.utf8()))}
    )
    df = nw.from_native(table, eager_only=True)
    result = df["a"].cat.get_categories().to_list()
    expected = ["a", "b", "d"]
    assert result == expected
