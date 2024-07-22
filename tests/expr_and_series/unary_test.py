from typing import Any

import narwhals as nw
from tests.utils import compare_dicts


def test_unary(constructor_with_lazy: Any) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    result = (
        nw.from_native(constructor_with_lazy(data))
        .with_columns(
            a_mean=nw.col("a").mean(),
            a_sum=nw.col("a").sum(),
            b_nunique=nw.col("b").n_unique(),
            z_min=nw.col("z").min(),
            z_max=nw.col("z").max(),
        )
        .select(nw.col("a_mean", "a_sum", "b_nunique", "z_min", "z_max").unique())
    )
    expected = {"a_mean": [2], "a_sum": [6], "b_nunique": [2], "z_min": [7], "z_max": [9]}
    compare_dicts(result, expected)
