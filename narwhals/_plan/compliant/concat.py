from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Protocol

from narwhals._plan.typing import NativeDataFrameT, NativeDataFrameT_co, NativeSeriesT
from narwhals._utils import _hasattr_static

if TYPE_CHECKING:
    from collections.abc import Iterable

    from typing_extensions import TypeIs

    from narwhals._plan.compliant.dataframe import CompliantDataFrame
    from narwhals._plan.compliant.series import CompliantSeries
    from narwhals.typing import ConcatMethod


class ConcatDataFrame(Protocol[NativeDataFrameT, NativeSeriesT]):
    def concat_df(
        self,
        dfs: Iterable[CompliantDataFrame[NativeDataFrameT, NativeSeriesT]],
        /,
        how: ConcatMethod,
    ) -> CompliantDataFrame[NativeDataFrameT, NativeSeriesT]:
        if how == "vertical":
            compliant = self.concat_df_vertical(dfs)
        elif how == "horizontal":
            compliant = self.concat_df_horizontal(dfs)
        else:
            compliant = self.concat_df_diagonal(dfs)
        return compliant

    # NOTE: Names adapted from from `polars._plr.pyi`
    def concat_df_vertical(
        self, dfs: Iterable[CompliantDataFrame[NativeDataFrameT, NativeSeriesT]], /
    ) -> CompliantDataFrame[NativeDataFrameT, NativeSeriesT]: ...
    def concat_df_diagonal(
        self, dfs: Iterable[CompliantDataFrame[NativeDataFrameT, NativeSeriesT]], /
    ) -> CompliantDataFrame[NativeDataFrameT, NativeSeriesT]: ...
    def concat_df_horizontal(
        self, dfs: Iterable[CompliantDataFrame[NativeDataFrameT, NativeSeriesT]], /
    ) -> CompliantDataFrame[NativeDataFrameT, NativeSeriesT]: ...


class ConcatSeries(Protocol[NativeSeriesT]):
    def concat_series(
        self, series: Iterable[CompliantSeries[NativeSeriesT]], /
    ) -> CompliantSeries[NativeSeriesT]: ...


class ConcatSeriesHorizontal(Protocol[NativeDataFrameT_co, NativeSeriesT]):
    def concat_series_horizontal(
        self, series: Iterable[CompliantSeries[NativeSeriesT]], /
    ) -> CompliantDataFrame[NativeDataFrameT_co, NativeSeriesT]: ...


def can_concat_dataframe(
    obj: ConcatDataFrame[NativeDataFrameT, NativeSeriesT] | Any,
) -> TypeIs[ConcatDataFrame[NativeDataFrameT, NativeSeriesT]]:
    return _hasattr_static(obj, "concat_df")
