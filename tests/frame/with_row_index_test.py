from typing import Any

import pytest

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts

data = {
    "a": ["foo", "bars"],
    "ab": ["foo", "bars"],
}


def test_with_row_index(request: pytest.FixtureRequest, constructor: Any) -> None:
    if "pyspark" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    result = nw.from_native(constructor(data)).with_row_index()
    expected = {"a": ["foo", "bars"], "ab": ["foo", "bars"], "index": [0, 1]}
    compare_dicts(result, expected)
