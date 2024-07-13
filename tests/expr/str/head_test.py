from typing import Any

import pytest

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts

data = {"a": ["foo", "bars"]}


def test_str_head(constructor: Any) -> None:
    if "pyarrow_table" in str(constructor):
        pytest.xfail()
    df = nw.from_native(constructor(data), eager_only=True)
    result = df.select(nw.col("a").str.head(3))
    expected = {
        "a": ["foo", "bar"],
    }
    compare_dicts(result, expected)
    result = df.select(df["a"].str.head(3))
    compare_dicts(result, expected)
