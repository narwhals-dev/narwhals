from typing import Any

import numpy as np

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts

data = {
    "a": ["foo", "bars"],
    "ab": ["foo", "bars"],
}


def test_with_columns(constructor_eager: Any) -> None:
    result = (
        nw.from_native(constructor_eager(data))
        .with_columns(d=np.array([4, 5]))
        .with_columns(e=nw.col("d") + 1)
        .select("d", "e")
    )
    expected = {"d": [4, 5], "e": [5, 6]}
    compare_dicts(result, expected)
