from typing import Any

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts

data = {
    "a": ["foo", "bars"],
    "ab": ["foo", "bars"],
}


def test_with_row_index(constructor_with_pyarrow: Any) -> None:
    result = nw.from_native(constructor_with_pyarrow(data)).with_row_index()
    expected = {"a": ["foo", "bars"], "ab": ["foo", "bars"], "index": [0, 1]}
    compare_dicts(result, expected)
    result = nw.from_native(constructor_with_pyarrow(data)).lazy().with_row_index()
    compare_dicts(result, expected)
