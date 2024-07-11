from typing import Any

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts


def test_count(constructor: Any) -> None:
    data = {"a": [1, 3, 2], "b": [4, None, 6], "z": [7.0, None, None]}
    df = nw.from_native(constructor(data), eager_only=True)
    result = df.select(nw.all().count())
    result_native = nw.to_native(result)
    expected = {"a": [3], "b": [2], "z": [1]}
    compare_dicts(result_native, expected)
