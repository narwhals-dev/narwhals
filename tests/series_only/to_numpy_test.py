from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import narwhals.stable.v1 as nw

if TYPE_CHECKING:
    from tests.utils import ConstructorEager


def test_to_numpy(
    constructor_eager: ConstructorEager, request: pytest.FixtureRequest
) -> None:
    if (
        "pandas_constructor" in str(constructor_eager)
        or "modin_constructor" in str(constructor_eager)
        or "cudf_constructor" in str(constructor_eager)
    ):
        request.applymarker(pytest.mark.xfail)

    data = [1, 2, None]
    s = nw.from_native(constructor_eager({"a": data}), eager_only=True)["a"].cast(
        nw.Int64
    )
    assert s.to_numpy().dtype == "float64"
    assert s.shape == (3,)

    assert_array_equal(s.to_numpy(), np.array(data, dtype=float))
