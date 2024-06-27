from __future__ import annotations

import pandas as pd
import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from pandas.testing import assert_frame_equal

import narwhals.stable.v1 as nw
from narwhals.utils import parse_version

pl_version = parse_version(pl.__version__)
pd_version = parse_version(pd.__version__)


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
    st.lists(
        st.floats(),
        min_size=3,
        max_size=3,
    ),
    st.lists(
        st.sampled_from(["a", "b", "c"]),
        min_size=1,
        max_size=3,
        unique=True,
    ),
)  # type: ignore[misc]
@pytest.mark.skipif(pl_version < parse_version("0.20.13"), reason="0.0 == -0.0")
@pytest.mark.skipif(pd_version < parse_version("2.0.0"), reason="requires pyarrow")
@pytest.mark.slow()
def test_join(  # pragma: no cover
    integers: st.SearchStrategy[list[int]],
    other_integers: st.SearchStrategy[list[int]],
    floats: st.SearchStrategy[list[float]],
    cols: st.SearchStrategy[list[str]],
) -> None:
    data = {"a": integers, "b": other_integers, "c": floats}

    df_polars = pl.DataFrame(data)
    df_polars2 = pl.DataFrame(data)
    df_pl = nw.from_native(df_polars, eager_only=True)
    other_pl = nw.from_native(df_polars2, eager_only=True)
    dframe_pl = df_pl.join(other_pl, left_on=cols, right_on=cols, how="inner")

    df_pandas = pd.DataFrame(data)
    df_pandas2 = pd.DataFrame(data)
    df_pd = nw.from_native(df_pandas, eager_only=True)
    other_pd = nw.from_native(df_pandas2, eager_only=True)
    dframe_pd = df_pd.join(other_pd, left_on=cols, right_on=cols, how="inner")

    dframe_pd1 = nw.to_native(dframe_pl).to_pandas()
    dframe_pd1 = dframe_pd1.sort_values(
        by=dframe_pd1.columns.to_list(), ignore_index=True
    )

    dframe_pd2 = nw.to_native(dframe_pd)
    dframe_pd2 = dframe_pd2.sort_values(
        by=dframe_pd2.columns.to_list(), ignore_index=True
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
)  # type: ignore[misc]
@pytest.mark.slow()
@pytest.mark.skipif(pd_version < parse_version("2.0.0"), reason="requires pyarrow")
def test_cross_join(  # pragma: no cover
    integers: st.SearchStrategy[list[int]],
    other_integers: st.SearchStrategy[list[int]],
) -> None:
    data = {"a": integers, "b": other_integers}

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
        by=dframe_pd1.columns.to_list(), ignore_index=True
    )

    dframe_pd2 = nw.to_native(dframe_pd)
    dframe_pd2 = dframe_pd2.sort_values(
        by=dframe_pd2.columns.to_list(), ignore_index=True
    )

    assert_frame_equal(dframe_pd1, dframe_pd2)
