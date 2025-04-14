from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import Constructor
from tests.utils import assert_equal_data

data = {
    "a": ["foo", "bars"],
    "ab": ["foo", "bars"],
}


def test_with_row_index(constructor: Constructor, request: pytest.FixtureRequest) -> None:
    if (
        ("pyspark" in str(constructor))
        or "duckdb" in str(constructor)
        or "ibis" in str(constructor)
    ):
        request.applymarker(pytest.mark.xfail)
    result = nw.from_native(constructor(data)).with_row_index()
    expected = {"index": [0, 1], "a": ["foo", "bars"], "ab": ["foo", "bars"]}
    assert_equal_data(result, expected)
