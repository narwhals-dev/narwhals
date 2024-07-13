from typing import Any

import numpy as np
import pyarrow as pa
import pytest

import narwhals.stable.v1 as nw
from narwhals.utils import parse_version


def test_array_dunder(constructor_series: Any) -> None:
    if "pyarrow_series" in str(constructor_series) and parse_version(
        pa.__version__
    ) < parse_version("16.0.0"):  # pragma: no cover
        pytest.xfail()

    s = nw.from_native(constructor_series([1, 2, 3]), series_only=True)
    result = s.__array__(object)
    np.testing.assert_array_equal(result, np.array([1, 2, 3], dtype=object))
