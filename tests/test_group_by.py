from __future__ import annotations

import pandas as pd
import polars as pl
import pytest

import narwhals as nw
from tests.utils import compare_dicts

df_pandas = pd.DataFrame({"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]})
df_lazy = pl.LazyFrame({"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]})


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
