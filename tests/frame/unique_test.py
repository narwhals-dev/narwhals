from __future__ import annotations

from typing import Any

import pytest

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts


def test_unique(constructor_with_lazy: Any) -> None:
    if "pyarrow_table" in str(constructor_with_lazy):
        pytest.xfail()
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df_raw = constructor_with_lazy(data)
    df = nw.from_native(df_raw).lazy()
    result = nw.to_native(df.unique("b").sort("b"))
    expected = {"a": [1, 2], "b": [4, 6], "z": [7.0, 9.0]}
    compare_dicts(result, expected)
    result = nw.to_native(df.collect().unique("b").sort("b"))
    expected = {"a": [1, 2], "b": [4, 6], "z": [7.0, 9.0]}
    compare_dicts(result, expected)
