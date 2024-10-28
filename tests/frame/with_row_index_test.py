from __future__ import annotations

import narwhals.stable.v1 as nw
from tests.utils import Constructor
from tests.utils import assert_equal_data

data = {
    "a": ["foo", "bars"],
    "ab": ["foo", "bars"],
}


def test_with_row_index(constructor: Constructor) -> None:
    result = nw.from_native(constructor(data)).with_row_index()
    expected = {"a": ["foo", "bars"], "ab": ["foo", "bars"], "index": [0, 1]}
    assert_equal_data(result, expected)
