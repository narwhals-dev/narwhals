from typing import Any

import pytest

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts


def test_na(constructor: Any, request: Any) -> None:
    data_na = {"a": [None, 3, 2], "b": [4, 4, 6], "z": [7.0, None, 9]}
    if "dask" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor(data_na)).lazy()
    result_nna = df.filter((~nw.col("a").is_null()) & (~df.collect()["z"].is_null()))
    expected = {"a": [2], "b": [6], "z": [9]}
    compare_dicts(result_nna, expected)
