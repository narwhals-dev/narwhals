from typing import Any

import numpy as np

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts

data = {"a": [1, 1, 2]}


def test_unique_expr(constructor: Any) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(nw.col("a").unique())
    expected = {"a": [1, 2]}
    compare_dicts(result, expected)


def test_unique_series(constructor_eager: Any) -> None:
    series = nw.from_native(constructor_eager(data), eager_only=True)["a"]
    result = series.unique()
    expected = np.array([1, 2])
    assert (result.to_numpy() == expected).all()
