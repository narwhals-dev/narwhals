from __future__ import annotations

from typing import Any

import pandas as pd
import polars as pl
import pytest

import narwhals as nw
from tests.utils import compare_dicts

data_1 = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
data_2 = {"a": [1, 1, 3], "b": [4, 4, 6], "c": [7, 8, 9]}

df_pandas = pd.DataFrame(data_1)
df_lazy = pl.LazyFrame(data_1)


def test_group_by_complex() -> None:
    df = nw.LazyFrame(df_pandas)
    with pytest.warns(UserWarning, match="complex group-by"):
        result = nw.to_native(
            df.group_by("a").agg((nw.col("b") - nw.col("z").mean()).mean()).sort("a")
        )
    expected = {"a": [1, 2, 3], "b": [-3.0, -3.0, -4.0]}
    compare_dicts(result, expected)

    df = nw.LazyFrame(df_lazy)
    result = nw.to_native(
        df.group_by("a").agg((nw.col("b") - nw.col("z").mean()).mean()).sort("a")
    )
    expected = {"a": [1, 2, 3], "b": [-3.0, -3.0, -4.0]}
    compare_dicts(result, expected)


def test_invalid_group_by() -> None:
    df = nw.LazyFrame(df_pandas)
    with pytest.raises(RuntimeError, match="does your"):
        df.group_by("a").agg(nw.col("b"))
    with pytest.raises(
        ValueError, match=r"Anonymous expressions are not supported in group_by\.agg"
    ):
        df.group_by("a").agg(nw.all().mean())


@pytest.mark.parametrize("constructor", [pd.DataFrame, pl.DataFrame])
def test_group_by_iter(constructor: Any) -> None:
    df = nw.from_native(constructor(data_2), eager_only=True)
    expected_keys = [(1,), (3,)]
    keys = []
    for key, sub_df in df.group_by("a"):
        if key == (1,):
            expected = {"a": [1, 1], "b": [4, 4], "c": [7, 8]}
            compare_dicts(sub_df, expected)
            assert isinstance(sub_df, nw.DataFrame)
        keys.append(key)
    assert sorted(keys) == sorted(expected_keys)
    expected_keys = [(1, 4), (3, 6)]  # type: ignore[list-item]
    keys = []
    for key, _df in df.group_by("a", "b"):
        keys.append(key)
    assert sorted(keys) == sorted(expected_keys)
    keys = []
    for key, _df in df.group_by(["a", "b"]):
        keys.append(key)
    assert sorted(keys) == sorted(expected_keys)


@pytest.mark.parametrize("constructor", [pd.DataFrame, pl.DataFrame])
def test_group_by_len(constructor: Any) -> None:
    result = (
        nw.from_native(constructor(data_2)).group_by("a").agg(nw.col("b").len()).sort("a")
    )
    expected = {"a": [1, 3], "b": [2, 1]}
    compare_dicts(result, expected)
