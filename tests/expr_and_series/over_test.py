from typing import Any

import pytest

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts

data = {
    "a": ["a", "a", "b", "b", "b"],
    "b": [1, 2, 3, 5, 3],
    "c": [5, 4, 3, 2, 1],
}


def test_over_single(constructor: Any, request: pytest.FixtureRequest) -> None:
    if "pyspark" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor(data))
    result = df.with_columns(c_max=nw.col("c").max().over("a"))
    expected = {
        "a": ["a", "a", "b", "b", "b"],
        "b": [1, 2, 3, 5, 3],
        "c": [5, 4, 3, 2, 1],
        "c_max": [5, 5, 3, 3, 3],
    }
    compare_dicts(result, expected)


def test_over_multiple(constructor: Any, request: pytest.FixtureRequest) -> None:
    if "pyspark" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor(data))
    result = df.with_columns(c_min=nw.col("c").min().over("a", "b"))
    expected = {
        "a": ["a", "a", "b", "b", "b"],
        "b": [1, 2, 3, 5, 3],
        "c": [5, 4, 3, 2, 1],
        "c_min": [5, 4, 1, 2, 1],
    }
    compare_dicts(result, expected)


def test_over_invalid(constructor: Any, request: pytest.FixtureRequest) -> None:
    if "pyspark" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    if "polars" in str(constructor):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor(data))
    with pytest.raises(ValueError, match="Anonymous expressions"):
        df.with_columns(c_min=nw.all().min().over("a", "b"))
