from typing import Any

import numpy as np

import narwhals as nw


def test_array_dunder(constructor_series_with_pyarrow: Any) -> None:
    s = nw.from_native(constructor_series_with_pyarrow([1, 2, 3]), series_only=True)
    result = s.__array__(object)
    np.testing.assert_array_equal(result, np.array([1, 2, 3], dtype=object))
