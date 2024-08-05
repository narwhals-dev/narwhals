from typing import Any

import pytest

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts


def test_transform(constructor: Any, request: Any) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    if "dask" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor(data))
    result = df.with_columns(a=nw.col("a").is_between(-1, 1), b=nw.col("b").is_in([4, 5]))
    expected = {"a": [True, False, False], "b": [True, True, False], "z": [7, 8, 9]}
    compare_dicts(result, expected)
