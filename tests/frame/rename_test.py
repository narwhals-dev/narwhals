from typing import Any

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts


def test_rename(constructor_with_pyarrow: Any) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df = nw.from_native(constructor_with_pyarrow(data), eager_only=True)
    result = df.rename({"a": "x", "b": "y"})
    expected = {"x": [1, 3, 2], "y": [4, 4, 6], "z": [7.0, 8, 9]}
    compare_dicts(result, expected)
