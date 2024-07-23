from typing import Any

import numpy as np
import pytest

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts

data = {"a": [1, 3, 2], "b": [0, 2, -1]}


@pytest.mark.parametrize(
    ("descending", "expected"),
    [
        (True, {"a": [3, 2, 1], "b": [0, 2, -1]}),
        (False, {"a": [1, 2, 3], "b": [0, 2, -1]}),
    ],
)
def test_sort_expr(constructor: Any, descending: Any, expected: Any) -> None:
    df = nw.from_native(constructor(data), eager_only=True)
    result = df.select(nw.col("a").sort(descending=descending), "b")
    compare_dicts(result, expected)


@pytest.mark.parametrize(
    ("descending", "expected"), [(True, [3, 2, 1]), (False, [1, 2, 3])]
)
def test_sort_series(constructor_series: Any, descending: Any, expected: Any) -> None:
    series = nw.from_native(constructor_series(data["a"]), series_only=True)
    result = series.sort(descending=descending)
    assert (result.to_numpy() == np.array(expected)).all()
