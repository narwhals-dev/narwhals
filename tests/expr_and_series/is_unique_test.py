from typing import Any

import numpy as np
import pytest

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts

data = {
    "a": [1, 1, 2],
    "b": [1, 2, 3],
}


def test_is_unique_expr(constructor: Any, request: Any) -> None:
    if "modin" in str(constructor):
        # TODO(unassigned): why is Modin failing here?
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor(data))
    result = df.select(nw.all().is_unique())
    expected = {
        "a": [False, False, True],
        "b": [True, True, True],
    }
    compare_dicts(result, expected)


def test_is_unique_series(constructor_eager: Any) -> None:
    series = nw.from_native(constructor_eager(data), eager_only=True)["a"]
    result = series.is_unique()
    expected = np.array([False, False, True])
    assert (result.to_numpy() == expected).all()
