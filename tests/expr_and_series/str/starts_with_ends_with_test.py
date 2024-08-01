from __future__ import annotations

from typing import Any

import pytest

import narwhals.stable.v1 as nw

# Don't move this into typechecking block, for coverage
# purposes
from tests.utils import compare_dicts

data = {"a": ["fdas", "edfas"]}


def test_ends_with(constructor: Any, request: Any) -> None:
    if "dask" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor(data)).lazy()
    result = df.select(nw.col("a").str.ends_with("das"))
    expected = {
        "a": [True, False],
    }
    compare_dicts(result, expected)

    result = df.select(df.collect()["a"].str.ends_with("das"))
    expected = {
        "a": [True, False],
    }
    compare_dicts(result, expected)


def test_starts_with(constructor: Any, request: Any) -> None:
    if "dask" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor(data)).lazy()
    result = df.select(nw.col("a").str.starts_with("fda"))
    expected = {
        "a": [True, False],
    }
    compare_dicts(result, expected)

    result = df.select(df.collect()["a"].str.starts_with("fda"))
    expected = {
        "a": [True, False],
    }
    compare_dicts(result, expected)
