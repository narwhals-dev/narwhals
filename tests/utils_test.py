from __future__ import annotations

import re
import string
from typing import TYPE_CHECKING

import hypothesis.strategies as st
import pandas as pd
import polars as pl
import pytest
from hypothesis import given
from pandas.testing import assert_frame_equal
from pandas.testing import assert_index_equal
from pandas.testing import assert_series_equal

import narwhals.stable.v1 as nw
from narwhals.exceptions import ColumnNotFoundError
from narwhals.utils import check_column_exists
from narwhals.utils import parse_version
from tests.utils import PANDAS_VERSION
from tests.utils import get_module_version_as_tuple

if TYPE_CHECKING:
    from narwhals.series import Series
    from narwhals.typing import IntoSeriesT


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


@pytest.mark.parametrize(
    "column_names",
    ["b", ["a", "b"]],
)
def test_maybe_set_index_pandas_column_names(
    column_names: str | list[str] | None,
) -> None:
    df = nw.from_native(pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
    result = nw.maybe_set_index(df, column_names)
    expected = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}).set_index(column_names)
    assert_frame_equal(nw.to_native(result), expected)


@pytest.mark.parametrize(
    "column_names",
    [
        "b",
        ["a", "b"],
    ],
)
def test_maybe_set_index_polars_column_names(
    column_names: str | list[str] | None,
) -> None:
    df = nw.from_native(pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
    result = nw.maybe_set_index(df, column_names)
    assert result is df


@pytest.mark.parametrize(
    "native_df_or_series",
    [pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}), pd.Series([0, 1, 2])],
)
@pytest.mark.parametrize(
    ("narwhals_index", "pandas_index"),
    [
        (nw.from_native(pd.Series([1, 2, 0]), series_only=True), pd.Series([1, 2, 0])),
        (
            [
                nw.from_native(pd.Series([0, 1, 2]), series_only=True),
                nw.from_native(pd.Series([1, 2, 0]), series_only=True),
            ],
            [
                pd.Series([0, 1, 2]),
                pd.Series([1, 2, 0]),
            ],
        ),
    ],
)
def test_maybe_set_index_pandas_direct_index(
    narwhals_index: Series[IntoSeriesT] | list[Series[IntoSeriesT]] | None,
    pandas_index: pd.Series | list[pd.Series] | None,
    native_df_or_series: pd.DataFrame | pd.Series,
) -> None:
    df = nw.from_native(native_df_or_series, allow_series=True)
    result = nw.maybe_set_index(df, index=narwhals_index)
    if isinstance(native_df_or_series, pd.Series):
        native_df_or_series.index = pandas_index
        assert_series_equal(nw.to_native(result), native_df_or_series)
    else:
        expected = native_df_or_series.set_index(pandas_index)
        assert_frame_equal(nw.to_native(result), expected)


@pytest.mark.parametrize(
    "index",
    [
        nw.from_native(pd.Series([1, 2, 0]), series_only=True),
        [
            nw.from_native(pd.Series([0, 1, 2]), series_only=True),
            nw.from_native(pd.Series([1, 2, 0]), series_only=True),
        ],
    ],
)
def test_maybe_set_index_polars_direct_index(
    index: Series[IntoSeriesT] | list[Series[IntoSeriesT]] | None,
) -> None:
    df = nw.from_native(pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
    result = nw.maybe_set_index(df, index=index)
    assert result is df


def test_maybe_set_index_pandas_series_column_names() -> None:
    df = nw.from_native(pd.Series([0, 1, 2]), allow_series=True)
    with pytest.raises(
        ValueError, match="Cannot set index using column names on a Series"
    ):
        nw.maybe_set_index(df, column_names=["a"])


def test_maybe_set_index_pandas_either_index_or_column_names() -> None:
    df = nw.from_native(pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
    column_names = ["a", "b"]
    index = nw.from_native(pd.Series([0, 1, 2]), series_only=True)
    with pytest.raises(
        ValueError, match="Only one of `column_names` or `index` should be provided"
    ):
        nw.maybe_set_index(df, column_names=column_names, index=index)
    with pytest.raises(
        ValueError, match="Either `column_names` or `index` should be provided"
    ):
        nw.maybe_set_index(df)


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
    pandas_df = nw.from_native(pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
    result = nw.maybe_reset_index(pandas_df)
    expected = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    assert_frame_equal(nw.to_native(result), expected)
    assert result.to_native() is pandas_df.to_native()
    pandas_series = nw.from_native(
        pd.Series([1, 2, 3], index=[7, 8, 9]), series_only=True
    )
    result_s = nw.maybe_reset_index(pandas_series)
    expected_s = pd.Series([1, 2, 3], index=[0, 1, 2])
    assert_series_equal(nw.to_native(result_s), expected_s)
    pandas_series = nw.from_native(pd.Series([1, 2, 3]), series_only=True)
    result_s = nw.maybe_reset_index(pandas_series)
    expected_s = pd.Series([1, 2, 3])
    assert_series_equal(nw.to_native(result_s), expected_s)
    assert result_s.to_native() is pandas_series.to_native()


def test_maybe_reset_index_polars() -> None:
    df = nw.from_native(pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
    result = nw.maybe_reset_index(df)
    assert result is df
    series = nw.from_native(pl.Series([1, 2, 3]), series_only=True)
    result_s = nw.maybe_reset_index(series)
    assert result_s is series


@pytest.mark.skipif(PANDAS_VERSION < (1, 0, 0), reason="too old for convert_dtypes")
def test_maybe_convert_dtypes_pandas() -> None:
    import numpy as np

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


def test_get_trivial_version_with_uninstalled_module() -> None:
    result = get_module_version_as_tuple("non_existent_module")
    assert result == (0, 0, 0)


@given(n_bytes=st.integers(1, 100))  # type: ignore[misc]
@pytest.mark.slow
def test_generate_temporary_column_name(n_bytes: int) -> None:
    columns = ["abc", "XYZ"]

    temp_col_name = nw.generate_temporary_column_name(n_bytes=n_bytes, columns=columns)
    assert temp_col_name not in columns


def test_generate_temporary_column_name_raise() -> None:
    from itertools import product

    columns = [
        "".join(t)
        for t in product(
            string.ascii_lowercase + string.digits,
            string.ascii_lowercase + string.digits,
        )
    ]

    with pytest.raises(
        AssertionError,
        match="Internal Error: Narwhals was not able to generate a column name with ",
    ):
        nw.generate_temporary_column_name(n_bytes=1, columns=columns)


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("2020.1.2", (2020, 1, 2)),
        ("2020.1.2-dev123", (2020, 1, 2)),
        ("3.0.0.dev0+618.gb552dc95c9", (3, 0, 0)),
    ],
)
def test_parse_version(version: str, expected: tuple[int, ...]) -> None:
    assert parse_version(version) == expected


def test_check_column_exists() -> None:
    columns = ["a", "b", "c"]
    subset = ["d", "f"]
    with pytest.raises(
        ColumnNotFoundError,
        match=re.escape("Column(s) ['d', 'f'] not found in ['a', 'b', 'c']"),
    ):
        check_column_exists(columns, subset)
