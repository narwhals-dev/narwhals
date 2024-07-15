from typing import Any

import pytest

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts


def test_sum_all(request: Any, constructor: Any) -> None:
    if "pyarrow_table" in str(constructor):
        request.applymarker(pytest.mark.xfail)

    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df = nw.from_native(constructor(data), eager_only=True)
    result = df.select(nw.all().sum())
    expected = {"a": [6], "b": [14], "z": [24.0]}
    compare_dicts(result, expected)
