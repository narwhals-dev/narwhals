from typing import Any

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts


def test_filter(constructor: Any) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df = nw.from_native(constructor(data))
    result = df.filter(nw.col("a") > 1)
    expected = {"a": [3, 2], "b": [4, 6], "z": [8.0, 9.0]}
    compare_dicts(result, expected)


def test_filter_series(constructor_eager: Any) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df = nw.from_native(constructor_eager(data), eager_only=True).with_columns(
        mask=nw.col("a") > 1
    )
    result = df.filter(df["mask"]).drop("mask")
    expected = {"a": [3, 2], "b": [4, 6], "z": [8.0, 9.0]}
    compare_dicts(result, expected)
