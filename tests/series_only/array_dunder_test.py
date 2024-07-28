from typing import Any

import numpy as np
import pyarrow as pa
import pytest

import narwhals.stable.v1 as nw
from narwhals.utils import parse_version


def test_array_dunder(request: Any, constructor: Any) -> None:
    if "pyarrow_table" in str(constructor) and parse_version(
        pa.__version__
    ) < parse_version("16.0.0"):  # pragma: no cover
        request.applymarker(pytest.mark.xfail)

    s = nw.from_native(constructor({"a": [1, 2, 3]}), eager_only=True)["a"]
    result = s.__array__(object)
    np.testing.assert_array_equal(result, np.array([1, 2, 3], dtype=object))
