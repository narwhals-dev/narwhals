from __future__ import annotations

from functools import reduce
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Iterable

from narwhals import dtypes
from narwhals._pandas_like.dataframe import PandasDataFrame
from narwhals._pandas_like.expr import PandasExpr
from narwhals._pandas_like.series import PandasSeries
from narwhals._pandas_like.utils import horizontal_concat
from narwhals._pandas_like.utils import parse_into_exprs
from narwhals._pandas_like.utils import series_from_iterable
from narwhals._pandas_like.utils import vertical_concat
from narwhals.utils import flatten

if TYPE_CHECKING:
    from narwhals._pandas_like.typing import IntoPandasExpr


class PandasNamespace:
    Int64 = dtypes.Int64
    Int32 = dtypes.Int32
    Int16 = dtypes.Int16
    Int8 = dtypes.Int8
    UInt64 = dtypes.UInt64
    UInt32 = dtypes.UInt32
    UInt16 = dtypes.UInt16
    UInt8 = dtypes.UInt8
    Float64 = dtypes.Float64
    Float32 = dtypes.Float32
    Boolean = dtypes.Boolean
    String = dtypes.String
    Datetime = dtypes.Datetime

    def make_native_series(self, name: str, data: list[Any], index: Any) -> Any:
        if self._implementation == "pandas":
            import pandas as pd

            return pd.Series(name=name, data=data, index=index)
        if self._implementation == "modin":  # pragma: no cover
            import modin.pandas as mpd

            return mpd.Series(name=name, data=data, index=index)
        if self._implementation == "cudf":  # pragma: no cover
            import cudf

            return cudf.Series(name=name, data=data, index=index)
        raise NotImplementedError  # pragma: no cover

    # --- not in spec ---
    def __init__(self, implementation: str) -> None:
        self._implementation = implementation

    def _create_expr_from_callable(  # noqa: PLR0913
        self,
        func: Callable[[PandasDataFrame], list[PandasSeries]],
        *,
        depth: int,
        function_name: str,
        root_names: list[str] | None,
        output_names: list[str] | None,
    ) -> PandasExpr:
        return PandasExpr(
            func,
            depth=depth,
            function_name=function_name,
            root_names=root_names,
            output_names=output_names,
            implementation=self._implementation,
        )

    def _create_series_from_scalar(
        self, value: Any, series: PandasSeries
    ) -> PandasSeries:
        return PandasSeries(
            series_from_iterable(
                [value],
                name=series._series.name,
                index=series._series.index[0:1],
                implementation=self._implementation,
            ),
            implementation=self._implementation,
        )

    def _create_expr_from_series(self, series: PandasSeries) -> PandasExpr:
        return PandasExpr(
            lambda _df: [series],
            depth=0,
            function_name="series",
            root_names=None,
            output_names=None,
            implementation=self._implementation,
        )

    # --- selection ---
    def col(self, *column_names: str | Iterable[str]) -> PandasExpr:
        return PandasExpr.from_column_names(
            *flatten(column_names), implementation=self._implementation
        )

    def all(self) -> PandasExpr:
        return PandasExpr(
            lambda df: [
                PandasSeries(
                    df._dataframe.loc[:, column_name],
                    implementation=self._implementation,
                )
                for column_name in df.columns
            ],
            depth=0,
            function_name="all",
            root_names=None,
            output_names=None,
            implementation=self._implementation,
        )

    # --- reduction ---
    def sum(self, *column_names: str) -> PandasExpr:
        return PandasExpr.from_column_names(
            *column_names, implementation=self._implementation
        ).sum()

    def mean(self, *column_names: str) -> PandasExpr:
        return PandasExpr.from_column_names(
            *column_names, implementation=self._implementation
        ).mean()

    def max(self, *column_names: str) -> PandasExpr:
        return PandasExpr.from_column_names(
            *column_names, implementation=self._implementation
        ).max()

    def min(self, *column_names: str) -> PandasExpr:
        return PandasExpr.from_column_names(
            *column_names, implementation=self._implementation
        ).min()

    def len(self) -> PandasExpr:
        return PandasExpr(
            lambda df: [
                PandasSeries(
                    series_from_iterable(
                        [len(df._dataframe)],
                        name="len",
                        index=[0],
                        implementation=self._implementation,
                    ),
                    implementation=self._implementation,
                ),
            ],
            depth=0,
            function_name="len",
            root_names=None,
            output_names=["len"],
            implementation=self._implementation,
        )

    # --- horizontal ---
    def sum_horizontal(
        self, *exprs: IntoPandasExpr | Iterable[IntoPandasExpr]
    ) -> PandasExpr:
        return reduce(lambda x, y: x + y, parse_into_exprs(self._implementation, *exprs))

    def all_horizontal(
        self, *exprs: IntoPandasExpr | Iterable[IntoPandasExpr]
    ) -> PandasExpr:
        # Why is this showing up as uncovered? It defo is?
        return reduce(
            lambda x, y: x & y, parse_into_exprs(self._implementation, *exprs)
        )  # pragma: no cover

    def concat(
        self,
        items: Iterable[PandasDataFrame],
        *,
        how: str = "vertical",
    ) -> PandasDataFrame:
        dfs: list[Any] = [item._dataframe for item in items]
        if how == "horizontal":
            return PandasDataFrame(
                horizontal_concat(dfs, implementation=self._implementation),
                implementation=self._implementation,
            )
        if how == "vertical":
            return PandasDataFrame(
                vertical_concat(dfs, implementation=self._implementation),
                implementation=self._implementation,
            )
        raise NotImplementedError
