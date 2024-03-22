from __future__ import annotations

import os
import warnings
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest

import narwhals as nw
from tests.utils import compare_dicts

df_pandas = pd.DataFrame({"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]})
df_polars = pl.DataFrame({"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]})
df_lazy = pl.LazyFrame({"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]})
df_pandas_na = pd.DataFrame({"a": [None, 3, 2], "b": [4, 4, 6], "z": [7.0, None, 9]})
df_lazy_na = pl.LazyFrame({"a": [None, 3, 2], "b": [4, 4, 6], "z": [7.0, None, 9]})

if os.environ.get("CI", None):
    import modin.pandas as mpd

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        df_mpd = mpd.DataFrame(
            pd.DataFrame({"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]})
        )
else:
    df_mpd = df_pandas.copy()


@pytest.mark.parametrize(
    "df_raw",
    [df_pandas, df_polars, df_lazy],
)
def test_sort(df_raw: Any) -> None:
    df = nw.LazyFrame(df_raw)
    result = df.sort("a", "b")
    result_native = nw.to_native(result)
    expected = {
        "a": [1, 2, 3],
        "b": [4, 6, 4],
        "z": [7.0, 9.0, 8.0],
    }
    compare_dicts(result_native, expected)


@pytest.mark.parametrize(
    "df_raw",
    [df_pandas, df_lazy],
)
def test_filter(df_raw: Any) -> None:
    df = nw.LazyFrame(df_raw)
    result = df.filter(nw.col("a") > 1)
    result_native = nw.to_native(result)
    expected = {"a": [3, 2], "b": [4, 6], "z": [8.0, 9.0]}
    compare_dicts(result_native, expected)


@pytest.mark.parametrize(
    "df_raw",
    [df_pandas, df_polars],
)
def test_filter_series(df_raw: Any) -> None:
    df = nw.DataFrame(df_raw).with_columns(mask=nw.col("a") > 1)
    result = df.filter(df["mask"]).drop("mask")
    result_native = nw.to_native(result)
    expected = {"a": [3, 2], "b": [4, 6], "z": [8.0, 9.0]}
    compare_dicts(result_native, expected)


@pytest.mark.parametrize(
    "df_raw",
    [df_pandas, df_lazy],
)
def test_add(df_raw: Any) -> None:
    df = nw.LazyFrame(df_raw)
    result = df.with_columns(
        c=nw.col("a") + nw.col("b"),
        d=nw.col("a") - nw.col("a").mean(),
    )
    result_native = nw.to_native(result)
    expected = {
        "a": [1, 3, 2],
        "b": [4, 4, 6],
        "z": [7.0, 8.0, 9.0],
        "c": [5, 7, 8],
        "d": [-1.0, 1.0, 0.0],
    }
    compare_dicts(result_native, expected)


@pytest.mark.parametrize(
    "df_raw",
    [df_pandas, df_lazy],
)
def test_double(df_raw: Any) -> None:
    df = nw.LazyFrame(df_raw)
    result = df.with_columns(nw.all() * 2)
    result_native = nw.to_native(result)
    expected = {"a": [2, 6, 4], "b": [8, 8, 12], "z": [14.0, 16.0, 18.0]}
    compare_dicts(result_native, expected)


@pytest.mark.parametrize(
    "df_raw",
    [df_pandas, df_lazy],
)
def test_select(df_raw: Any) -> None:
    df = nw.LazyFrame(df_raw)
    result = df.select("a")
    result_native = nw.to_native(result)
    expected = {"a": [1, 3, 2]}
    compare_dicts(result_native, expected)


@pytest.mark.parametrize("df_raw", [df_pandas, df_lazy])
def test_sumh(df_raw: Any) -> None:
    df = nw.LazyFrame(df_raw)
    result = df.with_columns(horizonal_sum=nw.sum_horizontal(nw.col("a"), nw.col("b")))
    result_native = nw.to_native(result)
    expected = {
        "a": [1, 3, 2],
        "b": [4, 4, 6],
        "z": [7.0, 8.0, 9.0],
        "horizonal_sum": [5, 7, 8],
    }
    compare_dicts(result_native, expected)


@pytest.mark.parametrize("df_raw", [df_pandas, df_lazy])
def test_sumh_literal(df_raw: Any) -> None:
    df = nw.LazyFrame(df_raw)
    result = df.with_columns(horizonal_sum=nw.sum_horizontal("a", nw.col("b")))
    result_native = nw.to_native(result)
    expected = {
        "a": [1, 3, 2],
        "b": [4, 4, 6],
        "z": [7.0, 8.0, 9.0],
        "horizonal_sum": [5, 7, 8],
    }
    compare_dicts(result_native, expected)


@pytest.mark.parametrize("df_raw", [df_pandas, df_lazy])
def test_sum_all(df_raw: Any) -> None:
    df = nw.LazyFrame(df_raw)
    result = df.select(nw.all().sum())
    result_native = nw.to_native(result)
    expected = {"a": [6], "b": [14], "z": [24.0]}
    compare_dicts(result_native, expected)


@pytest.mark.parametrize("df_raw", [df_pandas, df_lazy])
def test_double_selected(df_raw: Any) -> None:
    df = nw.LazyFrame(df_raw)
    result = df.select(nw.col("a", "b") * 2)
    result_native = nw.to_native(result)
    expected = {"a": [2, 6, 4], "b": [8, 8, 12]}
    compare_dicts(result_native, expected)


@pytest.mark.parametrize("df_raw", [df_pandas, df_lazy])
def test_rename(df_raw: Any) -> None:
    df = nw.LazyFrame(df_raw)
    result = df.rename({"a": "x", "b": "y"})
    result_native = nw.to_native(result)
    expected = {"x": [1, 3, 2], "y": [4, 4, 6], "z": [7.0, 8, 9]}
    compare_dicts(result_native, expected)


@pytest.mark.parametrize("df_raw", [df_pandas, df_lazy])
def test_join(df_raw: Any) -> None:
    df = nw.LazyFrame(df_raw)
    df_right = df.rename({"z": "z_right"})
    result = df.join(df_right, left_on=["a", "b"], right_on=["a", "b"], how="inner")
    result_native = nw.to_native(result)
    expected = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9], "z_right": [7.0, 8, 9]}
    compare_dicts(result_native, expected)


@pytest.mark.parametrize("df_raw", [df_pandas, df_lazy])
def test_schema(df_raw: Any) -> None:
    df = nw.LazyFrame(df_raw)
    result = df.schema
    expected = {"a": nw.dtypes.Int64, "b": nw.dtypes.Int64, "z": nw.dtypes.Float64}
    assert result == expected


@pytest.mark.parametrize("df_raw", [df_pandas, df_lazy])
def test_columns(df_raw: Any) -> None:
    df = nw.LazyFrame(df_raw)
    result = df.columns
    expected = ["a", "b", "z"]
    assert result == expected


@pytest.mark.parametrize("df_raw", [df_polars, df_pandas, df_mpd, df_lazy])
def test_lazy_instantiation(df_raw: Any) -> None:
    result = nw.LazyFrame(df_raw)
    result_native = nw.to_native(result)
    expected = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    compare_dicts(result_native, expected)


@pytest.mark.parametrize("df_raw", [df_lazy])
def test_lazy_instantiation_error(df_raw: Any) -> None:
    with pytest.raises(
        TypeError, match="Can't instantiate DataFrame from Polars LazyFrame."
    ):
        _ = nw.DataFrame(df_raw).shape


@pytest.mark.parametrize("df_raw", [df_polars, df_pandas, df_mpd])
def test_eager_instantiation(df_raw: Any) -> None:
    result = nw.DataFrame(df_raw)
    result_native = nw.to_native(result)
    expected = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    compare_dicts(result_native, expected)


def test_accepted_dataframes() -> None:
    array = np.array([[0, 4.0], [2, 5]])
    with pytest.raises(
        TypeError,
        match="Expected pandas-like dataframe, Polars dataframe, or Polars lazyframe, got: <class 'numpy.ndarray'>",
    ):
        nw.DataFrame(array)
    with pytest.raises(
        TypeError,
        match="Expected pandas-like dataframe, Polars dataframe, or Polars lazyframe, got: <class 'numpy.ndarray'>",
    ):
        nw.LazyFrame(array)


@pytest.mark.parametrize("df_raw", [df_polars, df_pandas, df_mpd])
@pytest.mark.filterwarnings("ignore:.*Passing a BlockManager.*:DeprecationWarning")
def test_convert_pandas(df_raw: Any) -> None:
    result = nw.DataFrame(df_raw).to_pandas()
    expected = pd.DataFrame({"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]})
    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize("df_raw", [df_polars, df_pandas, df_mpd])
@pytest.mark.filterwarnings(
    r"ignore:np\.find_common_type is deprecated\.:DeprecationWarning"
)
def test_convert_numpy(df_raw: Any) -> None:
    result = nw.DataFrame(df_raw).to_numpy()
    expected = np.array([[1, 3, 2], [4, 4, 6], [7.0, 8, 9]]).T
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("df_raw", [df_polars, df_pandas, df_mpd])
def test_shape(df_raw: Any) -> None:
    result = nw.DataFrame(df_raw).shape
    expected = (3, 3)
    assert result == expected


@pytest.mark.parametrize("df_raw", [df_polars, df_pandas, df_mpd, df_lazy])
def test_expr_binary(df_raw: Any) -> None:
    result = nw.LazyFrame(df_raw).with_columns(
        a=(1 + 3 * nw.col("a")) * (1 / nw.col("a")),
        b=nw.col("z") / (2 - nw.col("b")),
        c=nw.col("a") + nw.col("b") / 2,
        d=nw.col("a") - nw.col("b"),
        e=((nw.col("a") > nw.col("b")) & (nw.col("a") >= nw.col("z"))).cast(nw.Int64),
        f=(
            (nw.col("a") < nw.col("b"))
            | (nw.col("a") <= nw.col("z"))
            | (nw.col("a") == 1)
        ).cast(nw.Int64),
    )
    result_native = nw.to_native(result)
    expected = {
        "a": [4, 3.333333, 3.5],
        "b": [-3.5, -4.0, -2.25],
        "z": [7.0, 8.0, 9.0],
        "c": [3, 5, 5],
        "d": [-3, -1, -4],
        "e": [0, 0, 0],
        "f": [1, 1, 1],
    }
    compare_dicts(result_native, expected)


@pytest.mark.parametrize("df_raw", [df_polars, df_pandas, df_lazy])
def test_expr_unary(df_raw: Any) -> None:
    result = (
        nw.LazyFrame(df_raw)
        .with_columns(
            a_mean=nw.col("a").mean(),
            a_sum=nw.col("a").sum(),
            b_nunique=nw.col("b").n_unique(),
            z_min=nw.col("z").min(),
            z_max=nw.col("z").max(),
        )
        .select(nw.col("a_mean", "a_sum", "b_nunique", "z_min", "z_max").unique())
    )
    result_native = nw.to_native(result)
    expected = {"a_mean": [2], "a_sum": [6], "b_nunique": [2], "z_min": [7], "z_max": [9]}
    compare_dicts(result_native, expected)


@pytest.mark.parametrize("df_raw", [df_polars, df_pandas, df_mpd, df_lazy])
def test_expr_transform(df_raw: Any) -> None:
    result = nw.LazyFrame(df_raw).with_columns(
        a=nw.col("a").is_between(-1, 1), b=nw.col("b").is_in([4, 5])
    )
    result_native = nw.to_native(result)
    expected = {"a": [True, False, False], "b": [True, True, False], "z": [7, 8, 9]}
    compare_dicts(result_native, expected)


@pytest.mark.parametrize("df_raw", [df_polars, df_pandas, df_lazy])
def test_expr_min_max(df_raw: Any) -> None:
    df = nw.LazyFrame(df_raw)
    result_min = nw.to_native(df.select(nw.min("a", "b", "z")))
    result_max = nw.to_native(df.select(nw.max("a", "b", "z")))
    expected_min = {"a": [1], "b": [4], "z": [7]}
    expected_max = {"a": [3], "b": [6], "z": [9]}
    compare_dicts(result_min, expected_min)
    compare_dicts(result_max, expected_max)


@pytest.mark.parametrize("df_raw", [df_polars, df_pandas, df_mpd, df_lazy])
def test_expr_sample(df_raw: Any) -> None:
    df = nw.LazyFrame(df_raw)
    result_shape = nw.to_native(df.select(nw.col("a").sample(n=2)).collect()).shape
    expected = (2, 1)
    assert result_shape == expected


@pytest.mark.parametrize("df_raw", [df_pandas_na, df_lazy_na])
def test_expr_na(df_raw: Any) -> None:
    df = nw.LazyFrame(df_raw)
    result_nna = nw.to_native(
        df.filter((~nw.col("a").is_null()) & (~nw.col("z").is_null()))
    )
    expected = {"a": [2], "b": [6], "z": [9]}
    compare_dicts(result_nna, expected)


@pytest.mark.parametrize("df_raw", [df_pandas, df_lazy])
def test_head(df_raw: Any) -> None:
    df = nw.LazyFrame(df_raw)
    result = nw.to_native(df.head(2))
    expected = {"a": [1, 3], "b": [4, 4], "z": [7.0, 8.0]}
    compare_dicts(result, expected)


@pytest.mark.parametrize("df_raw", [df_pandas, df_lazy])
def test_unique(df_raw: Any) -> None:
    df = nw.LazyFrame(df_raw)
    result = nw.to_native(df.unique("b").sort("b"))
    expected = {"a": [1, 2], "b": [4, 6], "z": [7.0, 9.0]}
    compare_dicts(result, expected)


@pytest.mark.parametrize("df_raw", [df_pandas_na, df_lazy_na])
def test_drop_nulls(df_raw: Any) -> None:
    df = nw.LazyFrame(df_raw)
    result = nw.to_native(df.select(nw.col("a").drop_nulls()))
    expected = {"a": [3, 2]}
    compare_dicts(result, expected)
