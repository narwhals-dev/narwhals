from __future__ import annotations

import narwhals as nw
from tests.utils import ConstructorEager, assert_equal_data


def test_tail(nw_eager_constructor: ConstructorEager) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8.0, 9.0]}
    expected = {"a": [3, 2], "b": [4, 6], "z": [8.0, 9]}
    df_raw = nw_eager_constructor(data)
    df = nw.from_native(df_raw)
    result = df.tail(2)
    assert_equal_data(result, expected)
    result = df.tail(-1)
    assert_equal_data(result, expected)
