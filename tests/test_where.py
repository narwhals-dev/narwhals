from __future__ import annotations

from typing import Any

import pytest

import narwhals.stable.v1 as nw
from narwhals.expression import when
from tests.utils import compare_dicts

data = {
    "a": [1, 1, 2],
    "b": ["a", "b", "c"],
    "c": [4.1, 5.0, 6.0],
    "d": [True, False, True],
}


def test_when(request: Any, constructor: Any) -> None:
    if "pyarrow_table" in str(constructor):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor(data))
    result = df.with_columns(when(nw.col("a") == 1).then(value=3).alias("a_when"))
    expected = {
        "a": [1, 1, 2],
        "b": ["a", "b", "c"],
        "c": [4.1, 5.0, 6.0],
        "d": [True, False, True],
        "a_when": [3, 3, None],
    }
    compare_dicts(result, expected)


def test_when_otherwise(request: Any, constructor: Any) -> None:
    if "pyarrow_table" in str(constructor):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor(data))
    result = df.with_columns(when(nw.col("a") == 1).then(3).otherwise(6).alias("a_when"))
    expected = {
        "a": [1, 1, 2],
        "b": ["a", "b", "c"],
        "c": [4.1, 5.0, 6.0],
        "d": [True, False, True],
        "a_when": [3, 3, 6],
    }
    compare_dicts(result, expected)
