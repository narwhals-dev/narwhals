from typing import Any

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts

data = {
    "a": [1.0, 2.0, None, 4.0],
    "b": [None, 3.0, None, 5.0],
}


def test_drop_nulls(constructor_with_pyarrow: Any) -> None:
    result = nw.from_native(constructor_with_pyarrow(data)).drop_nulls()
    expected = {
        "a": [2.0, 4.0],
        "b": [3.0, 5.0],
    }
    compare_dicts(result, expected)
    result = nw.from_native(constructor_with_pyarrow(data)).lazy().drop_nulls()
    compare_dicts(result, expected)
