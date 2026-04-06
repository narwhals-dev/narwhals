from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from typing_extensions import assert_type

import narwhals as nw
from narwhals.typing import IntoFrameT


def test_readme_example() -> None:
    # check that readme example (as of March 2026) passes
    def _agnostic_function(  # pragma: no cover
        df_native: IntoFrameT, date_column: str, price_column: str
    ) -> IntoFrameT:
        df = nw.from_native(df_native)
        assert_type(df, nw.DataFrame[IntoFrameT] | nw.LazyFrame[IntoFrameT])
        res = (
            df.group_by(nw.col(date_column).dt.truncate("1mo"))
            .agg(nw.col(price_column).mean())
            .sort(date_column)
        )
        assert_type(res, nw.DataFrame[IntoFrameT] | nw.LazyFrame[IntoFrameT])
        native = res.to_native()
        assert_type(native, IntoFrameT)
        return res.to_native()


def test_from_eager_or_lazy_polars() -> None:
    pytest.importorskip("polars")
    import polars as pl

    lf = pl.LazyFrame()
    df = pl.DataFrame()
    either = lf if df.height else df

    r_lf = nw.from_native(lf)
    r_df = nw.from_native(df)
    r_either = nw.from_native(either)

    r2_lf = nw.from_native(r_lf)
    r2_df = nw.from_native(r_df)
    r2_either = nw.from_native(r_either)

    if TYPE_CHECKING:
        assert_type(r_lf, nw.LazyFrame[pl.LazyFrame])
        assert_type(r_df, nw.DataFrame[pl.DataFrame])
        assert_type(r_either, nw.DataFrame[pl.DataFrame] | nw.LazyFrame[pl.LazyFrame])
        assert_type(r2_lf, nw.LazyFrame[pl.LazyFrame])
        assert_type(r2_df, nw.DataFrame[pl.DataFrame])
        assert_type(r2_either, nw.DataFrame[pl.DataFrame] | nw.LazyFrame[pl.LazyFrame])


def test_into_frame_t_incompatible_apis() -> None:
    def _agnostic_function(  # pragma: no cover
        df_native: IntoFrameT,
    ) -> IntoFrameT:
        nw.from_native(df_native).sink_parquet("...")  # type: ignore[union-attr]
        _ = (
            nw.from_native(df_native)
            .with_row_index()  # type: ignore[call-arg]
            .to_native()
        )
        return (
            nw.from_native(df_native)
            .unique(maintain_order=True)  # type: ignore[call-arg]
            .to_native()
        )
