"""Smoke-test for the fireducks import hook.

Fireducks is intended as a drop-in replacement for pandas, so we take their word for it
and just check that some basic operations are supported.

Run this test with `python -m fireducks.pandas --hook-import-module -m pytest`.
"""

from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import assert_equal_data

pytest.importorskip("pandas")
import pandas as pd


def test_from_native_dataframe() -> None:
    df = pd.DataFrame({"a": [1, 2, 3]})
    result = nw.from_native(df).with_columns(b=nw.col("a") * 2)
    assert isinstance(result, nw.DataFrame)
    assert_equal_data(result, {"a": [1, 2, 3], "b": [2, 4, 6]})


def test_from_native_series() -> None:
    s = pd.Series([1, 2, -3])
    result = nw.from_native(s, series_only=True).abs()
    assert isinstance(result, nw.Series)
    assert_equal_data({"a": result}, {"a": [1, 2, 3]})
