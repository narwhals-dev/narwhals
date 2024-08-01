from typing import Any

import pytest

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts


def test_clip(request: Any, constructor: Any) -> None:
    if "pyarrow_table" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor({"a": [1, 2, 3, -4, 5]}))
    result = df.select(b=nw.col("a").clip(3, 5))
    expected = {"b": [3, 3, 3, 3, 5]}
    compare_dicts(result, expected)


def test_clip_series(request: Any, constructor_eager: Any) -> None:
    if "pyarrow_table" in str(constructor_eager):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor_eager({"a": [1, 2, 3, -4, 5]}), eager_only=True)
    result = {"b": df["a"].clip(3, 5)}
    expected = {"b": [3, 3, 3, 3, 5]}
    compare_dicts(result, expected)
