from __future__ import annotations

import narwhals._plan as nwp
from tests.plan.utils import assert_equal_data, dataframe


def test_clone() -> None:
    data_1 = {"a": [1, 2], "b": [3, 4]}
    data_2 = {"a": [1, 2], "b": [3, 4], "c": [4, 6]}
    df = dataframe(data_1)
    df_clone = df.clone()
    assert df is not df_clone
    assert df._compliant is not df_clone._compliant
    assert_equal_data(df_clone, data_1)
    df_clone_mod = df_clone.with_columns((nwp.col("a") + nwp.col("b")).alias("c"))
    assert_equal_data(df, data_1)
    assert_equal_data(df_clone, data_1)
    assert_equal_data(df_clone_mod, data_2)
