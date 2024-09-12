from typing import Any

import pytest

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts

data = {"a": ["foo", "bars"]}


def test_str_head(constructor: Any, request: pytest.FixtureRequest) -> None:
    if "pyspark" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor(data))
    result = df.select(nw.col("a").str.head(3))
    expected = {
        "a": ["foo", "bar"],
    }
    compare_dicts(result, expected)


def test_str_head_series(constructor_eager: Any) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    expected = {
        "a": ["foo", "bar"],
    }
    result = df.select(df["a"].str.head(3))
    compare_dicts(result, expected)
