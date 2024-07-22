from typing import Any

import numpy as np
import pytest

import narwhals.stable.v1 as nw

data = {"a": [4, 4, 6]}


def test_value_counts(request: Any, constructor: Any) -> None:
    if "pandas_series_nullable" in str(constructor):  # fails for py3.8
        request.applymarker(pytest.mark.xfail)

    s = nw.from_native(constructor(data), eager_only=True)["a"]

    sorted_result = s.value_counts(sort=True)
    assert sorted_result.columns == ["a", "count"]

    expected = np.array([[4, 2], [6, 1]])
    assert (sorted_result.to_numpy() == expected).all()

    unsorted_result = s.value_counts(sort=False)
    assert unsorted_result.columns == ["a", "count"]

    a = unsorted_result.to_numpy()
    assert (a[a[:, 0].argsort()] == expected).all()
