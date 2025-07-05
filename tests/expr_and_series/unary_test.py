from __future__ import annotations

from contextlib import nullcontext as does_not_raise

import pytest

import narwhals as nw
from tests.utils import Constructor, ConstructorEager, assert_equal_data


def test_unary(constructor: Constructor, request: pytest.FixtureRequest) -> None:
    if "ibis" in str(constructor):
        request.applymarker(pytest.mark.xfail)

    data = {"a": [1, 3, 2], "b": [4, 4, 6], "c": [7.0, 8.0, None], "z": [7.0, 8.0, 9.0]}
    result = nw.from_native(constructor(data)).select(
        a_mean=nw.col("a").mean(),
        a_median=nw.col("a").median(),
        a_sum=nw.col("a").sum(),
        a_skew=nw.col("a").skew(),
        b_nunique=nw.col("b").n_unique(),
        b_skew=nw.col("b").skew(),
        c_nunique=nw.col("c").n_unique(),
        z_min=nw.col("z").min(),
        z_max=nw.col("z").max(),
    )
    expected = {
        "a_mean": [2],
        "a_median": [2],
        "a_sum": [6],
        "a_skew": [0.0],
        "b_nunique": [2],
        "b_skew": [0.7071067811865465],
        "c_nunique": [3],
        "z_min": [7],
        "z_max": [9],
    }
    assert_equal_data(result, expected)


def test_unary_series(constructor_eager: ConstructorEager) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "c": [7.0, 8.0, None], "z": [7.0, 8.0, 9.0]}
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = {
        "a_mean": [df["a"].mean()],
        "a_median": [df["a"].median()],
        "a_sum": [df["a"].sum()],
        "a_skew": [df["a"].skew()],
        "b_nunique": [df["b"].n_unique()],
        "b_skew": [df["b"].skew()],
        "c_nunique": [df["c"].n_unique()],
        "c_skew": [df["c"].skew()],
        "z_min": [df["z"].min()],
        "z_max": [df["z"].max()],
    }
    expected = {
        "a_mean": [2.0],
        "a_median": [2],
        "a_sum": [6],
        "a_skew": [0.0],
        "b_nunique": [2],
        "b_skew": [0.7071067811865465],
        "c_nunique": [3],
        "c_skew": [0.0],
        "z_min": [7.0],
        "z_max": [9.0],
    }
    assert_equal_data(result, expected)


def test_unary_two_elements(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if "ibis" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    data = {"a": [1, 2], "b": [2, 10], "c": [2.0, None]}
    result = nw.from_native(constructor(data)).select(
        a_nunique=nw.col("a").n_unique(),
        a_skew=nw.col("a").skew(),
        b_nunique=nw.col("b").n_unique(),
        b_skew=nw.col("b").skew(),
        c_nunique=nw.col("c").n_unique(),
        c_skew=nw.col("c").skew(),
    )
    expected = {
        "a_nunique": [2],
        "a_skew": [0.0],
        "b_nunique": [2],
        "b_skew": [0.0],
        "c_nunique": [2],
        "c_skew": [None],
    }
    assert_equal_data(result, expected)


def test_unary_two_elements_series(constructor_eager: ConstructorEager) -> None:
    data = {"a": [1, 2], "b": [2, 10], "c": [2.0, None]}
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = {
        "a_nunique": [df["a"].n_unique()],
        "a_skew": [df["a"].skew()],
        "b_nunique": [df["b"].n_unique()],
        "b_skew": [df["b"].skew()],
        "c_nunique": [df["c"].n_unique()],
        "c_skew": [df["c"].skew()],
    }
    expected = {
        "a_nunique": [2],
        "a_skew": [0.0],
        "b_nunique": [2],
        "b_skew": [0.0],
        "c_nunique": [2],
        "c_skew": [None],
    }
    assert_equal_data(result, expected)


def test_unary_one_element(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if "pyspark" in str(constructor) and "sqlframe" not in str(constructor):
        request.applymarker(pytest.mark.xfail)
    if "ibis" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    data = {"a": [1], "b": [2], "c": [None]}
    # Dask runs into a divide by zero RuntimeWarning for 1 element skew.
    context = (
        pytest.warns(RuntimeWarning, match="invalid value encountered in scalar divide")
        if "dask" in str(constructor)
        else does_not_raise()
    )

    result = (
        nw.from_native(constructor(data))
        .with_columns(nw.col("c").cast(nw.Float64))
        .select(
            a_nunique=nw.col("a").n_unique(),
            a_skew=nw.col("a").skew(),
            b_nunique=nw.col("b").n_unique(),
            b_skew=nw.col("b").skew(),
            c_nunique=nw.col("c").n_unique(),
            c_skew=nw.col("c").skew(),
        )
    )
    expected = {
        "a_nunique": [1],
        "a_skew": [None],
        "b_nunique": [1],
        "b_skew": [None],
        "c_nunique": [1],
        "c_skew": [None],
    }
    with context:
        assert_equal_data(result, expected)


def test_unary_one_element_series(constructor_eager: ConstructorEager) -> None:
    data = {"a": [1], "b": [2], "c": [None]}
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = {
        "a_nunique": [df["a"].n_unique()],
        "a_skew": [df["a"].skew()],
        "b_nunique": [df["b"].n_unique()],
        "b_skew": [df["b"].skew()],
        "c_nunique": [df["c"].n_unique()],
        "c_skew": [df["c"].skew()],
    }
    expected = {
        "a_nunique": [1],
        "a_skew": [None],
        "b_nunique": [1],
        "b_skew": [None],
        "c_nunique": [1],
        "c_skew": [None],
    }
    assert_equal_data(result, expected)
