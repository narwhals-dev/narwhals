from typing import Any

import numpy as np

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts

data = {
    "a": [1, 1, 2],
    "b": [1, 2, 3],
}


def test_is_duplicated_expr(constructor: Any) -> None:
    df = nw.from_native(constructor(data), eager_only=True)
    result = df.select(nw.all().is_duplicated())
    expected = {
        "a": [True, True, False],
        "b": [False, False, False],
    }
    compare_dicts(result, expected)


def test_is_duplicated_series(constructor_series: Any) -> None:
    series = nw.from_native(constructor_series(data["a"]), series_only=True)
    result = series.is_duplicated()
    expected = np.array([True, True, False])
    assert (result.to_numpy() == expected).all()
