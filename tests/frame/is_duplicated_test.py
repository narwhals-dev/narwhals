from __future__ import annotations

import narwhals as nw
from tests.utils import ConstructorEager, assert_equal_data


def test_is_duplicated(nw_eager_constructor: ConstructorEager) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8.0, 9.0]}
    df_raw = nw_eager_constructor(data)
    df = nw.from_native(df_raw, eager_only=True)
    result = nw.concat([df, df.head(1)]).is_duplicated()
    expected = {"is_duplicated": [True, False, False, True]}
    assert_equal_data({"is_duplicated": result}, expected)


def test_is_duplicated_with_nulls(nw_eager_constructor: ConstructorEager) -> None:
    data = {"col1": [1, 2, 3], "col2": ["one", None, None]}
    df_raw = nw_eager_constructor(data)
    df = nw.from_native(df_raw, eager_only=True)
    result = df.is_duplicated()
    expected = {"is_duplicated": [False, False, False]}
    assert_equal_data({"is_duplicated": result}, expected)
