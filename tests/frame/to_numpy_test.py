from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import narwhals.stable.v1 as nw

if TYPE_CHECKING:
    from tests.utils import ConstructorEager


def test_to_numpy(constructor_eager: ConstructorEager) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.1, 8, 9]}
    df_raw = constructor_eager(data)
    result = nw.from_native(df_raw, eager_only=True).to_numpy()

    expected = np.array([[1, 3, 2], [4, 4, 6], [7.1, 8, 9]]).T
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == "float64"
