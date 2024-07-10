from typing import Any

import numpy as np

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts

data = {
    "a": ["foo", "bars"],
    "ab": ["foo", "bars"],
}


def test_with_columns(constructor: Any) -> None:
    result = nw.from_native(constructor(data)).with_columns(d=np.array([4, 5]))
    expected = {"a": ["foo", "bars"], "ab": ["foo", "bars"], "d": [4, 5]}
    compare_dicts(result, expected)
