from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest
from hypothesis import assume, given, strategies as st
from pandas.testing import assert_frame_equal

import narwhals as nw
from tests.utils import PANDAS_VERSION, POLARS_VERSION, assert_equal_data

pytest.importorskip("pandas")
pytest.importorskip("polars")
pytest.importorskip("pyarrow")

import pandas as pd
import polars as pl
import pyarrow as pa

if TYPE_CHECKING:
    from collections.abc import Mapping


@given(
    st.lists(
        st.integers(min_value=-9223372036854775807, max_value=9223372036854775807),
        min_size=3,
        max_size=3,
    ),
    st.lists(
        st.integers(min_value=-9223372036854775807, max_value=9223372036854775807),
        min_size=3,
        max_size=3,
    ),
    st.lists(st.floats(), min_size=3, max_size=3),
    st.lists(st.sampled_from(["a", "b", "c"]), min_size=1, max_size=3, unique=True),
)
@pytest.mark.skipif(POLARS_VERSION < (0, 20, 13), reason="0.0 == -0.0")
@pytest.mark.skipif(PANDAS_VERSION < (2, 0, 0), reason="requires pyarrow")
@pytest.mark.slow
def test_join(  # pragma: no cover
    integers: st.SearchStrategy[list[int]],
    other_integers: st.SearchStrategy[list[int]],
    floats: st.SearchStrategy[list[float]],
    cols: st.SearchStrategy[list[str]],
) -> None:
    data: Mapping[str, Any] = {"a": integers, "b": other_integers, "c": floats}
    join_cols = cast("list[str]", cols)

    df_polars = pl.DataFrame(data)
    df_polars2 = pl.DataFrame(data)
    df_pl = nw.from_native(df_polars, eager_only=True)
    other_pl = nw.from_native(df_polars2, eager_only=True)

    dframe_pl = df_pl.join(other_pl, left_on=join_cols, right_on=join_cols, how="inner")

    df_pandas = pd.DataFrame(data)
    df_pandas2 = pd.DataFrame(data)
    df_pd = nw.from_native(df_pandas, eager_only=True)
    other_pd = nw.from_native(df_pandas2, eager_only=True)
    dframe_pd = df_pd.join(other_pd, left_on=join_cols, right_on=join_cols, how="inner")

    dframe_pd1 = nw.to_native(dframe_pl).to_pandas()
    dframe_pd1 = dframe_pd1.sort_values(
        by=dframe_pd1.columns.to_list(), ignore_index=True, inplace=False
    )

    dframe_pd2 = nw.to_native(dframe_pd)
    dframe_pd2 = dframe_pd2.sort_values(
        by=dframe_pd2.columns.to_list(), ignore_index=True, inplace=False
    )

    assert_frame_equal(dframe_pd1, dframe_pd2)


@given(
    st.lists(
        st.integers(min_value=-9223372036854775807, max_value=9223372036854775807),
        min_size=3,
        max_size=3,
    ),
    st.lists(
        st.integers(min_value=-9223372036854775807, max_value=9223372036854775807),
        min_size=3,
        max_size=3,
    ),
)
@pytest.mark.skipif(PANDAS_VERSION < (2, 0, 0), reason="requires pyarrow")
@pytest.mark.slow
def test_cross_join(  # pragma: no cover
    integers: st.SearchStrategy[list[int]], other_integers: st.SearchStrategy[list[int]]
) -> None:
    data: Mapping[str, Any] = {"a": integers, "b": other_integers}

    df_polars = pl.DataFrame(data)
    df_polars2 = pl.DataFrame(data)
    df_pl = nw.from_native(df_polars, eager_only=True)
    other_pl = nw.from_native(df_polars2, eager_only=True)
    dframe_pl = df_pl.join(other_pl, how="cross")

    df_pandas = pd.DataFrame(data)
    df_pandas2 = pd.DataFrame(data)
    df_pd = nw.from_native(df_pandas, eager_only=True)
    other_pd = nw.from_native(df_pandas2, eager_only=True)
    dframe_pd = df_pd.join(other_pd, how="cross")

    dframe_pd1 = nw.to_native(dframe_pl).to_pandas()
    dframe_pd1 = dframe_pd1.sort_values(
        by=dframe_pd1.columns.to_list(), ignore_index=True, inplace=False
    )

    dframe_pd2 = nw.to_native(dframe_pd)
    dframe_pd2 = dframe_pd2.sort_values(
        by=dframe_pd2.columns.to_list(), ignore_index=True, inplace=False
    )

    assert_frame_equal(dframe_pd1, dframe_pd2)


@given(
    a_left_data=st.lists(st.integers(min_value=0, max_value=5), min_size=3, max_size=3),
    b_left_data=st.lists(st.integers(min_value=0, max_value=5), min_size=3, max_size=3),
    c_left_data=st.lists(st.integers(min_value=0, max_value=5), min_size=3, max_size=3),
    a_right_data=st.lists(st.integers(min_value=0, max_value=5), min_size=3, max_size=3),
    b_right_data=st.lists(st.integers(min_value=0, max_value=5), min_size=3, max_size=3),
    d_right_data=st.lists(st.integers(min_value=0, max_value=5), min_size=3, max_size=3),
    left_key=st.lists(
        st.sampled_from(["a", "b", "c"]), min_size=1, max_size=3, unique=True
    ),
    right_key=st.lists(
        st.sampled_from(["a", "b", "d"]), min_size=1, max_size=3, unique=True
    ),
)
@pytest.mark.slow
def test_left_join(  # pragma: no cover
    a_left_data: list[int],
    b_left_data: list[int],
    c_left_data: list[int],
    a_right_data: list[int],
    b_right_data: list[int],
    d_right_data: list[int],
    left_key: list[str],
    right_key: list[str],
) -> None:
    assume(len(left_key) == len(right_key))
    data_left: dict[str, Any] = {"a": a_left_data, "b": b_left_data, "c": c_left_data}
    data_right: dict[str, Any] = {"a": a_right_data, "b": b_right_data, "d": d_right_data}
    result_pd = nw.from_native(pd.DataFrame(data_left), eager_only=True).join(
        nw.from_native(pd.DataFrame(data_right), eager_only=True),
        how="left",
        left_on=left_key,
        right_on=right_key,
    )
    result_pl = nw.to_native(
        nw.from_native(pl.DataFrame(data_left), eager_only=True).join(
            nw.from_native(pl.DataFrame(data_right), eager_only=True),
            how="left",
            left_on=left_key,
            right_on=right_key,
        )
    )
    assert_equal_data(
        result_pd.to_dict(as_series=False), result_pl.to_dict(as_series=False)
    )
    # For PyArrow, insert an extra sort, as the order of rows isn't guaranteed
    result_pa = (
        nw.from_native(pa.table(data_left), eager_only=True)
        .join(
            nw.from_native(pa.table(data_right), eager_only=True),
            how="left",
            left_on=left_key,
            right_on=right_key,
        )
        .select(nw.all().cast(nw.Float64))
        .pipe(lambda df: df.sort(df.columns))
    )
    assert_equal_data(
        result_pa, result_pd.pipe(lambda df: df.sort(df.columns)).to_dict(as_series=False)
    )
