from __future__ import annotations

import narwhals as nw
from tests.utils import ConstructorEager, assert_equal_data


def test_is_unique(constructor_eager: ConstructorEager) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8.0, 9.0]}
    df_raw = constructor_eager(data)
    df = nw.from_native(df_raw, eager_only=True)
    result = nw.concat([df, df.head(1)]).is_unique()
    expected = {"is_unique": [False, True, True, False]}
    assert_equal_data({"is_unique": result}, expected)


def test_is_unique_with_nulls(constructor_eager: ConstructorEager) -> None:
    data = {"col1": [1, 2, 3, 3], "col2": ["one", None, None, None]}
    df_raw = constructor_eager(data)
    df = nw.from_native(df_raw, eager_only=True)
    result = df.is_unique()
    expected = {"is_unique": [True, True, False, False]}
    assert_equal_data({"is_unique": result}, expected)
