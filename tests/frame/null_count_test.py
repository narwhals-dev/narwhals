from __future__ import annotations

import narwhals as nw
from tests.utils import ConstructorEager, assert_equal_data


def test_null_count(nw_eager_constructor: ConstructorEager) -> None:
    data = {"a": [None, 3, 2], "b": [4, 4, 6], "z": [7.0, None, 9]}
    df_raw = nw_eager_constructor(data)
    df = nw.from_native(df_raw, eager_only=True)
    result = df.null_count()
    expected = {"a": [1], "b": [0], "z": [1]}
    assert_equal_data(result, expected)
