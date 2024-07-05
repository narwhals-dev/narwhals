from typing import Any

import narwhals as nw
from tests.utils import compare_dicts


def test_select(constructor_with_pyarrow: Any) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df = nw.from_native(constructor_with_pyarrow(data), eager_only=True)
    result = df.select("a")
    result_native = nw.to_native(result)
    expected = {"a": [1, 3, 2]}
    compare_dicts(result_native, expected)
