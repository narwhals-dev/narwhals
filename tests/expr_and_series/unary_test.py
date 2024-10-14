from typing import Any

import pytest

import narwhals.stable.v1 as nw
from tests.utils import Constructor
from tests.utils import compare_dicts


def test_unary(constructor: Constructor, request: pytest.FixtureRequest) -> None:
    if "dask" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    result = (
        nw.from_native(constructor(data))
        .with_columns(
            a_mean=nw.col("a").mean(),
            a_sum=nw.col("a").sum(),
            a_skew=nw.col("a").skew(),
            b_nunique=nw.col("b").n_unique(),
            z_min=nw.col("z").min(),
            z_max=nw.col("z").max(),
        )
        .select(
            nw.col("a_mean", "a_sum", "a_skew", "b_nunique", "z_min", "z_max").unique()
        )
    )
    expected = {
        "a_mean": [2],
        "a_sum": [6],
        "a_skew": [0.0],
        "b_nunique": [2],
        "z_min": [7],
        "z_max": [9],
    }
    compare_dicts(result, expected)


def test_unary_series(constructor_eager: Any) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = {
        "a_mean": [df["a"].mean()],
        "a_sum": [df["a"].sum()],
        "a_skew": [df["a"].skew()],
        "b_nunique": [df["b"].n_unique()],
        "z_min": [df["z"].min()],
        "z_max": [df["z"].max()],
    }
    expected = {
        "a_mean": [2.0],
        "a_sum": [6],
        "a_skew": [0.0],
        "b_nunique": [2],
        "z_min": [7.0],
        "z_max": [9.0],
    }
    compare_dicts(result, expected)
