from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, Any

import pytest

from narwhals.dependencies import is_numpy_scalar

pytest.importorskip("numpy")

import numpy as np

if TYPE_CHECKING:
    from narwhals.typing import _NumpyScalar


@pytest.mark.parametrize(
    "data",
    [
        np.int64(-70),
        np.uint16(12),
        np.float64(94.999),
        np.str_("word"),
        np.datetime64(dt.date(2000, 1, 1)),
    ],
)
def test_is_numpy_scalar_valid(data: _NumpyScalar) -> None:
    assert is_numpy_scalar(data)


@pytest.mark.parametrize("data", [-70, 12, 94.999, "word", dt.date(2000, 1, 1)])
def test_is_numpy_scalar_invalid(data: Any) -> None:
    assert not is_numpy_scalar(data)
