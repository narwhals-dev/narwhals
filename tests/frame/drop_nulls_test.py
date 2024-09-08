from __future__ import annotations

from typing import Any

import pytest

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts

data = {
    "a": [1.0, 2.0, None, 4.0],
    "b": [None, 3.0, None, 5.0],
}


def test_drop_nulls(request: pytest.FixtureRequest, constructor: Any) -> None:
    if "pyspark" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    result = nw.from_native(constructor(data)).drop_nulls()
    expected = {
        "a": [2.0, 4.0],
        "b": [3.0, 5.0],
    }
    compare_dicts(result, expected)


@pytest.mark.parametrize("subset", ["a", ["a"]])
def test_drop_nulls_subset(
    request: pytest.FixtureRequest, constructor: Any, subset: str | list[str]
) -> None:
    if "pyspark" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    result = nw.from_native(constructor(data)).drop_nulls(subset=subset)
    expected = {
        "a": [1, 2.0, 4.0],
        "b": [float("nan"), 3.0, 5.0],
    }
    compare_dicts(result, expected)
