from __future__ import annotations

from typing import Literal

import pandas as pd
import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts
from tests.utils import is_windows


@given(
    integers=st.lists(
        st.integers(min_value=-9223372036854775807, max_value=9223372036854775807),
        min_size=3,
        max_size=3,
    ),
    other_integers=st.lists(
        st.integers(min_value=-9223372036854775807, max_value=9223372036854775807),
        min_size=3,
        max_size=3,
    ),
    floats=st.lists(
        st.floats(),
        min_size=3,
        max_size=3,
    ),
    how=st.sampled_from(["horizontal", "vertical"]),
)  # type: ignore[misc]
@pytest.mark.slow
@pytest.mark.skipif(is_windows(), reason="pyarrow breaking on windows")
def test_concat(  # pragma: no cover
    integers: list[int],
    other_integers: list[int],
    floats: list[float],
    how: Literal["horizontal", "vertical"],
) -> None:
    data = {"a": integers, "b": other_integers, "c": floats}

    df_polars = pl.DataFrame(data)
    df_polars2 = pl.DataFrame(data)
    df_pandas = pd.DataFrame(data)
    df_pandas2 = pd.DataFrame(data)

    if how == "horizontal":
        df_pl = nw.from_native(df_polars).rename({"a": "d", "b": "e"}).drop("c").lazy()
        df_pd = nw.from_native(df_pandas).rename({"a": "d", "b": "e"}).drop("c").lazy()
    else:
        df_pl = nw.from_native(df_polars, eager_only=True).lazy()
        df_pd = nw.from_native(df_pandas, eager_only=True).lazy()

    other_pl = nw.from_native(df_polars2, eager_only=True).lazy()
    dframe_pl = nw.concat([df_pl, other_pl], how=how)

    other_pd = nw.from_native(df_pandas2).lazy()
    dframe_pd = nw.concat([df_pd, other_pd], how=how)

    dframe_pd1 = nw.to_native(dframe_pl)
    dframe_pd2 = nw.to_native(dframe_pd)

    compare_dicts(dframe_pd1, dframe_pd2)
