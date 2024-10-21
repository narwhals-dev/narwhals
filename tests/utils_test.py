from __future__ import annotations

import pandas as pd
import polars as pl
import pytest
from pandas.testing import assert_frame_equal
from pandas.testing import assert_index_equal
from pandas.testing import assert_series_equal

import narwhals.stable.v1 as nw


def test_maybe_align_index_pandas() -> None:
    df = nw.from_native(pd.DataFrame({"a": [1, 2, 3]}, index=[1, 2, 0]))
    s = nw.from_native(pd.Series([1, 2, 3], index=[2, 1, 0]), series_only=True)
    result = nw.maybe_align_index(df, s)
    expected = pd.DataFrame({"a": [2, 1, 3]}, index=[2, 1, 0])
    assert_frame_equal(nw.to_native(result), expected)
    result = nw.maybe_align_index(df, df.sort("a", descending=True))
    expected = pd.DataFrame({"a": [3, 2, 1]}, index=[0, 2, 1])
    assert_frame_equal(nw.to_native(result), expected)
    result_s = nw.maybe_align_index(s, df)
    expected_s = pd.Series([2, 1, 3], index=[1, 2, 0])
    assert_series_equal(nw.to_native(result_s), expected_s)
    result_s = nw.maybe_align_index(s, s.sort(descending=True))
    expected_s = pd.Series([3, 2, 1], index=[0, 1, 2])
    assert_series_equal(nw.to_native(result_s), expected_s)


def test_with_columns_sort() -> None:
    # Check that, unlike in pandas, we don't change the index
    # when sorting
    df = nw.from_native(pd.DataFrame({"a": [2, 1, 3]}))
    result = df.with_columns(a_sorted=nw.col("a").sort()).pipe(nw.to_native)
    expected = pd.DataFrame({"a": [2, 1, 3], "a_sorted": [1, 2, 3]})
    assert_frame_equal(result, expected)


def test_non_unique_index() -> None:
    df = nw.from_native(pd.DataFrame({"a": [1, 2, 3]}, index=[1, 2, 0]))
    s = nw.from_native(pd.Series([1, 2, 3], index=[2, 2, 0]), series_only=True)
    with pytest.raises(ValueError, match="unique"):
        nw.maybe_align_index(df, s)


def test_maybe_align_index_polars() -> None:
    df = nw.from_native(pl.DataFrame({"a": [1, 2, 3]}))
    s = nw.from_native(pl.Series([1, 2, 3]), series_only=True)
    result = nw.maybe_align_index(df, s)
    assert result is df
    with pytest.raises(ValueError, match="length"):
        nw.maybe_align_index(df, s[1:])


def test_maybe_set_index_pandas() -> None:
    df = nw.from_native(pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}, index=[1, 2, 0]))
    result = nw.maybe_set_index(df, "b")
    expected = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}, index=[1, 2, 0]).set_index(
        "b"
    )
    assert_frame_equal(nw.to_native(result), expected)


def test_maybe_set_index_polars() -> None:
    df = nw.from_native(pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
    result = nw.maybe_set_index(df, "b")
    assert result is df


def test_maybe_get_index_pandas() -> None:
    pandas_df = pd.DataFrame({"a": [1, 2, 3]}, index=[1, 2, 0])
    result = nw.maybe_get_index(nw.from_native(pandas_df))
    assert_index_equal(result, pandas_df.index)
    pandas_series = pd.Series([1, 2, 3], index=[1, 2, 0])
    result_s = nw.maybe_get_index(nw.from_native(pandas_series, series_only=True))
    assert_index_equal(result_s, pandas_series.index)


def test_maybe_get_index_polars() -> None:
    df = nw.from_native(pl.DataFrame({"a": [1, 2, 3]}))
    result = nw.maybe_get_index(df)
    assert result is None
    series = nw.from_native(pl.Series([1, 2, 3]), series_only=True)
    result = nw.maybe_get_index(series)
    assert result is None


def test_maybe_reset_index_pandas() -> None:
    pandas_df = nw.from_native(
        pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}, index=[7, 8, 9])
    )
    result = nw.maybe_reset_index(pandas_df)
    expected = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}, index=[0, 1, 2])
    assert_frame_equal(nw.to_native(result), expected)
    pandas_series = nw.from_native(
        pd.Series([1, 2, 3], index=[7, 8, 9]), series_only=True
    )
    result_s = nw.maybe_reset_index(pandas_series)
    expected_s = pd.Series([1, 2, 3], index=[0, 1, 2])
    assert_series_equal(nw.to_native(result_s), expected_s)


def test_maybe_reset_index_polars() -> None:
    df = nw.from_native(pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
    result = nw.maybe_reset_index(df)
    assert result is df
    series = nw.from_native(pl.Series([1, 2, 3]), series_only=True)
    result_s = nw.maybe_reset_index(series)
    assert result_s is series


def test_maybe_convert_dtypes_pandas(
    request: pytest.FixtureRequest, pandas_version: tuple[int, ...]
) -> None:
    import numpy as np

    if pandas_version < (1, 0, 0):
        request.applymarker(pytest.mark.skip(reason="too old for convert_dtypes"))
    df = nw.from_native(
        pd.DataFrame({"a": [1, np.nan]}, dtype=np.dtype("float64")), eager_only=True
    )
    result = nw.to_native(nw.maybe_convert_dtypes(df))
    expected = pd.DataFrame({"a": [1, pd.NA]}, dtype="Int64")
    pd.testing.assert_frame_equal(result, expected)
    result_s = nw.to_native(nw.maybe_convert_dtypes(df["a"]))
    expected_s = pd.Series([1, pd.NA], name="a", dtype="Int64")
    pd.testing.assert_series_equal(result_s, expected_s)


def test_maybe_convert_dtypes_polars() -> None:
    import numpy as np

    df = nw.from_native(pl.DataFrame({"a": [1.1, np.nan]}))
    result = nw.maybe_convert_dtypes(df)
    assert result is df
