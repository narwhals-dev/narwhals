from __future__ import annotations

import narwhals as nw
from tests.utils import ConstructorEager, assert_equal_data


def test_clone(constructor_eager: ConstructorEager) -> None:
    expected = {"a": [1, 2], "b": [3, 4]}
    expected_mod = {"a": [1, 2], "b": [3, 4], "c": [4, 6]}
    df = nw.from_native(constructor_eager(expected), eager_only=True)
    df_clone = df.clone()
    assert df is not df_clone
    assert df._compliant_frame is not df_clone._compliant_frame
    assert_equal_data(df_clone, expected)
    df_clone_mod = df_clone.with_columns((nw.col("a") + nw.col("b")).alias("c"))
    assert_equal_data(df, expected)
    assert_equal_data(df_clone, expected)
    assert_equal_data(df_clone_mod, expected_mod)
