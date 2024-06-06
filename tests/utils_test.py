import pandas as pd
import polars as pl
import pytest
from pandas.testing import assert_frame_equal
from pandas.testing import assert_series_equal

import narwhals as nw


def test_maybe_align_index_pandas() -> None:
    df = nw.from_native(pd.DataFrame({"a": [1, 2, 3]}, index=[1, 2, 0]))
    s = nw.from_native(pd.Series([1, 2, 3], index=[2, 1, 0]), series_only=True)
    result = nw.maybe_align_index(df, s)
    expected = pd.DataFrame({"a": [2, 1, 3]}, index=[2, 1, 0])
    assert_frame_equal(nw.to_native(result), expected)
    result = nw.maybe_align_index(s, df)
    expected = pd.Series([2, 1, 3], index=[1, 2, 0])
    assert_series_equal(nw.to_native(result), expected)
    result = nw.maybe_align_index(s, s.sort(descending=True))
    expected = pd.Series([3, 2, 1], index=[0, 1, 2])
    assert_series_equal(nw.to_native(result), expected)
    result = nw.maybe_align_index(df, df.sort("a", descending=True))
    expected = pd.DataFrame({"a": [3, 2, 1]}, index=[0, 2, 1])
    assert_frame_equal(nw.to_native(result), expected)


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


def test_native_namespace() -> None:
    df = nw.from_native(pl.DataFrame({"a": [1, 2, 3]}))
    assert nw.get_native_namespace(df) is pl
    df = nw.from_native(pd.DataFrame({"a": [1, 2, 3]}))
    assert nw.get_native_namespace(df) is pd


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


def test_maybe_convert_dtypes_pandas() -> None:
    import numpy as np

    df = nw.from_native(pd.DataFrame({"a": [1, np.nan]}, dtype=np.dtype("float64")))
    result = nw.to_native(nw.maybe_convert_dtypes(df))
    expected = pd.DataFrame({"a": [1, pd.NA]}, dtype="Int64")
    pd.testing.assert_frame_equal(result, expected)


def test_maybe_convert_dtypes_polars() -> None:
    import numpy as np

    df = nw.from_native(pl.DataFrame({"a": [1.1, np.nan]}))
    result = nw.maybe_convert_dtypes(df)
    assert result is df
