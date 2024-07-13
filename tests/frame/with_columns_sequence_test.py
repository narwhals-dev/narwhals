from typing import Any

import numpy as np
import pytest

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts

data = {
    "a": ["foo", "bars"],
    "ab": ["foo", "bars"],
}


def test_with_columns(constructor_with_pyarrow: Any, request: Any) -> None:
    if "table" in str(constructor_with_pyarrow):
        request.applymarker(pytest.mark.xfail)

    result = (
        nw.from_native(constructor_with_pyarrow(data))
        .with_columns(d=np.array([4, 5]))
        .with_columns(d2=nw.col("d") ** 2)
    )
    expected = {"a": ["foo", "bars"], "ab": ["foo", "bars"], "d": [4, 5], "d2": [16, 25]}
    compare_dicts(result, expected)
