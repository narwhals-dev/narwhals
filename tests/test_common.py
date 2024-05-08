from __future__ import annotations

import os
import warnings
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest
from pandas.testing import assert_series_equal as pd_assert_series_equal
from polars.testing import assert_series_equal as pl_assert_series_equal

import narwhals as nw
from narwhals.utils import parse_version
from tests.utils import compare_dicts

df_pandas = pd.DataFrame({"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]})
if parse_version(pd.__version__) >= parse_version("1.5.0"):
    df_pandas_pyarrow = pd.DataFrame(
        {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    ).astype(
        {
            "a": "Int64[pyarrow]",
            "b": "Int64[pyarrow]",
            "z": "Float64[pyarrow]",
        }
    )
    df_pandas_nullable = pd.DataFrame(
        {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    ).astype(
        {
            "a": "Int64",
            "b": "Int64",
            "z": "Float64",
        }
    )
else:  # pragma: no cover
    df_pandas_pyarrow = df_pandas
    df_pandas_nullable = df_pandas
df_polars = pl.DataFrame({"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]})
df_lazy = pl.LazyFrame({"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]})
df_pandas_na = pd.DataFrame({"a": [None, 3, 2], "b": [4, 4, 6], "z": [7.0, None, 9]})
df_lazy_na = pl.LazyFrame({"a": [None, 3, 2], "b": [4, 4, 6], "z": [7.0, None, 9]})
df_right_pandas = pd.DataFrame({"c": [6, 12, -1], "d": [0, -4, 2]})
df_right_lazy = pl.LazyFrame({"c": [6, 12, -1], "d": [0, -4, 2]})

if os.environ.get("CI", None):
    try:
        import modin.pandas as mpd
    except ImportError:  # pragma: no cover
        df_mpd = df_pandas.copy()
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            df_mpd = mpd.DataFrame(
                pd.DataFrame({"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]})
            )
else:  # pragma: no cover
    df_mpd = df_pandas.copy()


@pytest.mark.parametrize(
    "df_raw",
    [df_pandas, df_polars, df_lazy, df_pandas_nullable, df_pandas_pyarrow],
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
    result = df.sort("a", "b", descending=[True, False])
    result_native = nw.to_native(result)
    expected = {
        "a": [3, 2, 1],
        "b": [4, 6, 4],
        "z": [8.0, 9.0, 7.0],
    }
    compare_dicts(result_native, expected)


@pytest.mark.parametrize(
    "df_raw",
    [df_pandas, df_lazy, df_pandas_nullable, df_pandas_pyarrow],
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
    [df_pandas, df_lazy, df_pandas_nullable, df_pandas_pyarrow],
)
def test_add(df_raw: Any) -> None:
    df = nw.LazyFrame(df_raw)
    result = df.with_columns(
        c=nw.col("a") + nw.col("b"),
        d=nw.col("a") - nw.col("a").mean(),
        e=nw.col("a") - nw.col("a").std(),
    )
    result_native = nw.to_native(result)
    expected = {
        "a": [1, 3, 2],
        "b": [4, 4, 6],
        "z": [7.0, 8.0, 9.0],
        "c": [5, 7, 8],
        "d": [-1.0, 1.0, 0.0],
        "e": [0.0, 2.0, 1.0],
    }
    compare_dicts(result_native, expected)


@pytest.mark.parametrize(
    "df_raw",
    [df_pandas, df_lazy, df_pandas_nullable, df_pandas_pyarrow],
)
def test_std(df_raw: Any) -> None:
    df = nw.LazyFrame(df_raw)
    result = df.select(
        nw.col("a").std().alias("a_ddof_default"),
        nw.col("a").std(ddof=1).alias("a_ddof_1"),
        nw.col("a").std(ddof=0).alias("a_ddof_0"),
        nw.col("b").std(ddof=2).alias("b_ddof_2"),
        nw.col("z").std(ddof=0).alias("z_ddof_0"),
    )
    result_native = nw.to_native(result)
    expected = {
        "a_ddof_default": [1.0],
        "a_ddof_1": [1.0],
        "a_ddof_0": [0.816497],
        "b_ddof_2": [1.632993],
        "z_ddof_0": [0.816497],
    }
    compare_dicts(result_native, expected)


@pytest.mark.parametrize(
    "df_raw",
    [df_pandas, df_lazy, df_pandas_nullable, df_pandas_pyarrow],
)
def test_double(df_raw: Any) -> None:
    df = nw.LazyFrame(df_raw)
    result = df.with_columns(nw.all() * 2)
    result_native = nw.to_native(result)
    expected = {"a": [2, 6, 4], "b": [8, 8, 12], "z": [14.0, 16.0, 18.0]}
    compare_dicts(result_native, expected)
    result = df.with_columns(nw.col("a").alias("o"), nw.all() * 2)
    result_native = nw.to_native(result)
    expected = {"o": [1, 3, 2], "a": [2, 6, 4], "b": [8, 8, 12], "z": [14.0, 16.0, 18.0]}
    compare_dicts(result_native, expected)


@pytest.mark.parametrize(
    "df_raw",
    [df_pandas, df_lazy, df_pandas_nullable, df_pandas_pyarrow],
)
def test_select(df_raw: Any) -> None:
    df = nw.LazyFrame(df_raw)
    result = df.select("a")
    result_native = nw.to_native(result)
    expected = {"a": [1, 3, 2]}
    compare_dicts(result_native, expected)


@pytest.mark.parametrize("df_raw", [df_pandas, df_lazy, df_pandas_nullable])
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


@pytest.mark.parametrize(
    "df_raw", [df_pandas, df_lazy, df_pandas_nullable, df_pandas_pyarrow]
)
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


@pytest.mark.parametrize(
    "df_raw", [df_pandas, df_lazy, df_pandas_nullable, df_pandas_pyarrow]
)
def test_sum_all(df_raw: Any) -> None:
    df = nw.LazyFrame(df_raw)
    result = df.select(nw.all().sum())
    result_native = nw.to_native(result)
    expected = {"a": [6], "b": [14], "z": [24.0]}
    compare_dicts(result_native, expected)


@pytest.mark.parametrize(
    "df_raw", [df_pandas, df_lazy, df_pandas_nullable, df_pandas_pyarrow]
)
def test_double_selected(df_raw: Any) -> None:
    df = nw.LazyFrame(df_raw)
    result = df.select(nw.col("a", "b") * 2)
    result_native = nw.to_native(result)
    expected = {"a": [2, 6, 4], "b": [8, 8, 12]}
    compare_dicts(result_native, expected)
    result = df.select("z", nw.col("a", "b") * 2)
    result_native = nw.to_native(result)
    expected = {"z": [7, 8, 9], "a": [2, 6, 4], "b": [8, 8, 12]}
    compare_dicts(result_native, expected)
    result = df.select("a").select(nw.col("a") + nw.all())
    result_native = nw.to_native(result)
    expected = {"a": [2, 6, 4]}
    compare_dicts(result_native, expected)


@pytest.mark.parametrize(
    "df_raw", [df_pandas, df_lazy, df_pandas_nullable, df_pandas_pyarrow]
)
def test_rename(df_raw: Any) -> None:
    df = nw.LazyFrame(df_raw)
    result = df.rename({"a": "x", "b": "y"})
    result_native = nw.to_native(result)
    expected = {"x": [1, 3, 2], "y": [4, 4, 6], "z": [7.0, 8, 9]}
    compare_dicts(result_native, expected)


@pytest.mark.parametrize(
    "df_raw", [df_pandas, df_lazy, df_pandas_nullable, df_pandas_pyarrow]
)
def test_join(df_raw: Any) -> None:
    df = nw.LazyFrame(df_raw)
    df_right = df
    result = df.join(df_right, left_on=["a", "b"], right_on=["a", "b"], how="inner")
    result_native = nw.to_native(result)
    expected = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9], "z_right": [7.0, 8, 9]}
    compare_dicts(result_native, expected)

    with pytest.raises(NotImplementedError):
        result = df.join(df_right, left_on="a", right_on="a", how="left")  # type: ignore[arg-type]

    result = df.collect().join(df_right.collect(), left_on="a", right_on="a", how="inner")  # type: ignore[assignment]
    result_native = nw.to_native(result)
    expected = {
        "a": [1, 3, 2],
        "b": [4, 4, 6],
        "b_right": [4, 4, 6],
        "z": [7.0, 8, 9],
        "z_right": [7.0, 8, 9],
    }
    compare_dicts(result_native, expected)


@pytest.mark.parametrize(
    "df_raw", [df_pandas, df_lazy, df_pandas_nullable, df_pandas_pyarrow]
)
def test_schema(df_raw: Any) -> None:
    result = nw.LazyFrame(df_raw).schema
    expected = {"a": nw.Int64, "b": nw.Int64, "z": nw.Float64}
    assert result == expected
    result = nw.LazyFrame(df_raw).collect().schema
    expected = {"a": nw.Int64, "b": nw.Int64, "z": nw.Float64}
    assert result == expected
    result = nw.LazyFrame(df_raw).columns  # type: ignore[assignment]
    expected = ["a", "b", "z"]  # type: ignore[assignment]
    assert result == expected
    result = nw.LazyFrame(df_raw).collect().columns  # type: ignore[assignment]
    expected = ["a", "b", "z"]  # type: ignore[assignment]
    assert result == expected


@pytest.mark.parametrize(
    "df_raw", [df_pandas, df_lazy, df_pandas_nullable, df_pandas_pyarrow]
)
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
    result = nw.from_native(df_raw).to_pandas()  # type: ignore[union-attr]
    expected = pd.DataFrame({"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]})
    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "df_raw", [df_polars, df_pandas, df_mpd, df_pandas_nullable, df_pandas_pyarrow]
)
@pytest.mark.filterwarnings(
    r"ignore:np\.find_common_type is deprecated\.:DeprecationWarning"
)
def test_convert_numpy(df_raw: Any) -> None:
    result = nw.DataFrame(df_raw).to_numpy()
    expected = np.array([[1, 3, 2], [4, 4, 6], [7.0, 8, 9]]).T
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == "float64"
    result = nw.DataFrame(df_raw).__array__()
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == "float64"


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
        g=nw.col("a") != 1,
        h=(False & (nw.col("a") != 1)),
        i=(False | (nw.col("a") != 1)),
        j=2 ** nw.col("a"),
        k=2 // nw.col("a"),
        l=nw.col("a") // 2,
        m=nw.col("a") ** 2,
        n=nw.col("a") % 2,
        o=2 % nw.col("a"),
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
        "g": [False, True, True],
        "h": [False, False, False],
        "i": [False, True, True],
        "j": [2, 8, 4],
        "k": [2, 0, 1],
        "l": [0, 1, 1],
        "m": [1, 9, 4],
        "n": [1, 1, 0],
        "o": [0, 2, 0],
    }
    compare_dicts(result_native, expected)


@pytest.mark.parametrize("df_raw", [df_polars, df_pandas, df_lazy])
def test_expr_unary(df_raw: Any) -> None:
    result = (
        nw.from_native(df_raw)
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
    result_shape = nw.to_native(df.collect()["a"].sample(n=2)).shape
    expected = (2,)  # type: ignore[assignment]
    assert result_shape == expected


@pytest.mark.parametrize("df_raw", [df_pandas_na, df_lazy_na])
def test_expr_na(df_raw: Any) -> None:
    df = nw.LazyFrame(df_raw)
    result_nna = nw.to_native(
        df.filter((~nw.col("a").is_null()) & (~df.collect()["z"].is_null()))
    )
    expected = {"a": [2], "b": [6], "z": [9]}
    compare_dicts(result_nna, expected)


@pytest.mark.parametrize(
    "df_raw", [df_pandas, df_lazy, df_pandas_nullable, df_pandas_pyarrow]
)
def test_head(df_raw: Any) -> None:
    df = nw.LazyFrame(df_raw)
    result = nw.to_native(df.head(2))
    expected = {"a": [1, 3], "b": [4, 4], "z": [7.0, 8.0]}
    compare_dicts(result, expected)
    result = nw.to_native(df.collect().head(2))
    expected = {"a": [1, 3], "b": [4, 4], "z": [7.0, 8.0]}
    compare_dicts(result, expected)


@pytest.mark.parametrize(
    "df_raw", [df_pandas, df_lazy, df_pandas_nullable, df_pandas_pyarrow]
)
def test_unique(df_raw: Any) -> None:
    df = nw.LazyFrame(df_raw)
    result = nw.to_native(df.unique("b").sort("b"))
    expected = {"a": [1, 2], "b": [4, 6], "z": [7.0, 9.0]}
    compare_dicts(result, expected)
    result = nw.to_native(df.collect().unique("b").sort("b"))
    expected = {"a": [1, 2], "b": [4, 6], "z": [7.0, 9.0]}
    compare_dicts(result, expected)


@pytest.mark.parametrize("df_raw", [df_pandas_na, df_lazy_na])
def test_drop_nulls(df_raw: Any) -> None:
    df = nw.LazyFrame(df_raw)
    result = nw.to_native(df.select(nw.col("a").drop_nulls()))
    expected = {"a": [3, 2]}
    compare_dicts(result, expected)
    result = nw.to_native(df.select(df.collect()["a"].drop_nulls()))
    expected = {"a": [3, 2]}
    compare_dicts(result, expected)


@pytest.mark.parametrize(
    ("df_raw", "df_raw_right"), [(df_pandas, df_right_pandas), (df_lazy, df_right_lazy)]
)
def test_concat_horizontal(df_raw: Any, df_raw_right: Any) -> None:
    df_left = nw.LazyFrame(df_raw)
    df_right = nw.LazyFrame(df_raw_right)
    result = nw.concat([df_left, df_right], how="horizontal")
    result_native = nw.to_native(result)
    expected = {
        "a": [1, 3, 2],
        "b": [4, 4, 6],
        "z": [7.0, 8, 9],
        "c": [6, 12, -1],
        "d": [0, -4, 2],
    }
    compare_dicts(result_native, expected)

    with pytest.raises(ValueError, match="No items"):
        nw.concat([])


@pytest.mark.parametrize(
    ("df_raw", "df_raw_right"), [(df_pandas, df_right_pandas), (df_lazy, df_right_lazy)]
)
def test_concat_vertical(df_raw: Any, df_raw_right: Any) -> None:
    df_left = nw.LazyFrame(df_raw).collect().rename({"a": "c", "b": "d"}).lazy().drop("z")
    df_right = nw.LazyFrame(df_raw_right)
    result = nw.concat([df_left, df_right], how="vertical")
    result_native = nw.to_native(result)
    expected = {"c": [1, 3, 2, 6, 12, -1], "d": [4, 4, 6, 0, -4, 2]}
    compare_dicts(result_native, expected)
    with pytest.raises(ValueError, match="No items"):
        nw.concat([], how="vertical")
    with pytest.raises(Exception, match="unable to vstack"):
        nw.concat([df_left, df_right.rename({"d": "i"})], how="vertical").collect()  # type: ignore[union-attr]


@pytest.mark.parametrize("df_raw", [df_pandas, df_polars])
def test_lazy(df_raw: Any) -> None:
    df = nw.DataFrame(df_raw)
    result = df.lazy()
    assert isinstance(result, nw.LazyFrame)


def test_to_dict() -> None:
    df = nw.DataFrame(df_pandas)
    result = df.to_dict(as_series=True)
    expected = {
        "a": pd.Series([1, 3, 2], name="a"),
        "b": pd.Series([4, 4, 6], name="b"),
        "z": pd.Series([7.0, 8, 9], name="z"),
    }
    for key in expected:
        pd_assert_series_equal(result[key], expected[key])

    df = nw.DataFrame(df_polars)
    result = df.to_dict(as_series=True)
    expected = {
        "a": pl.Series("a", [1, 3, 2]),
        "b": pl.Series("b", [4, 4, 6]),
        "z": pl.Series("z", [7.0, 8, 9]),
    }
    for key in expected:
        pl_assert_series_equal(result[key], expected[key])


@pytest.mark.parametrize(
    "df_raw", [df_pandas, df_lazy, df_pandas_nullable, df_pandas_pyarrow]
)
def test_any_all(df_raw: Any) -> None:
    df = nw.LazyFrame(df_raw)
    result = nw.to_native(df.select((nw.all() > 1).all()))
    expected = {"a": [False], "b": [True], "z": [True]}
    compare_dicts(result, expected)
    result = nw.to_native(df.select((nw.all() > 1).any()))
    expected = {"a": [True], "b": [True], "z": [True]}
    compare_dicts(result, expected)


def test_invalid() -> None:
    df = nw.LazyFrame(df_pandas)
    with pytest.raises(ValueError, match="Multi-output"):
        df.select(nw.all() + nw.all())
    with pytest.raises(TypeError, match="Perhaps you:"):
        df.select([pl.col("a")])  # type: ignore[list-item]
    with pytest.raises(TypeError, match="Perhaps you:"):
        df.select([nw.col("a").cast(pl.Int64)])


@pytest.mark.parametrize("df_raw", [df_pandas])
def test_reindex(df_raw: Any) -> None:
    df = nw.DataFrame(df_raw)
    result = df.select("b", df["a"].sort(descending=True))
    expected = {"b": [4, 4, 6], "a": [3, 2, 1]}
    compare_dicts(result, expected)
    result = df.select("b", nw.col("a").sort(descending=True))
    compare_dicts(result, expected)

    s = df["a"]
    result_s = s > s.sort()
    assert not result_s[0]
    assert result_s[1]
    assert not result_s[2]
    result = df.with_columns(s.sort())
    expected = {"a": [1, 2, 3], "b": [4, 4, 6], "z": [7.0, 8.0, 9.0]}  # type: ignore[list-item]
    compare_dicts(result, expected)
    with pytest.raises(ValueError, match="Multi-output expressions are not supported"):
        nw.to_native(df.with_columns(nw.all() + nw.all()))


@pytest.mark.parametrize(
    ("df_raw", "df_raw_right"),
    [(df_pandas, df_polars), (df_polars, df_pandas)],
)
def test_library(df_raw: Any, df_raw_right: Any) -> None:
    df_left = nw.LazyFrame(df_raw)
    df_right = nw.LazyFrame(df_raw_right)
    with pytest.raises(
        NotImplementedError, match="Cross-library comparisons aren't supported"
    ):
        nw.concat([df_left, df_right], how="horizontal")
    with pytest.raises(
        NotImplementedError, match="Cross-library comparisons aren't supported"
    ):
        nw.concat([df_left, df_right], how="vertical")
    with pytest.raises(
        NotImplementedError, match="Cross-library comparisons aren't supported"
    ):
        df_left.join(df_right, left_on=["a"], right_on=["a"], how="inner")
