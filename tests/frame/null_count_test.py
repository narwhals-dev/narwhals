from __future__ import annotations

import narwhals as nw
from tests.utils import ConstructorEager, assert_equal_data


def test_null_count(constructor_eager: ConstructorEager) -> None:
    data = {"a": [None, 3, 2], "b": [4, 4, 6], "z": [7.0, None, 9]}
    df_raw = constructor_eager(data)
    df = nw.from_native(df_raw, eager_only=True)
    result = df.null_count()
    expected = {"a": [1], "b": [0], "z": [1]}
    assert_equal_data(result, expected)
