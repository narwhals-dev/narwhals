from __future__ import annotations

from functools import reduce
from typing import Any
from typing import Callable
from typing import Iterable

from narwhals.pandas_like.dataframe import DataFrame
from narwhals.pandas_like.dataframe import LazyFrame
from narwhals.pandas_like.expr import Expr
from narwhals.pandas_like.series import Series
from narwhals.spec import AnyDataFrame
from narwhals.spec import DataFrame as DataFrameT
from narwhals.spec import Expr as ExprT
from narwhals.spec import IntoExpr
from narwhals.spec import LazyFrame as LazyFrameProtocol
from narwhals.spec import Namespace as NamespaceProtocol
from narwhals.spec import Series as SeriesT
from narwhals.utils import flatten_str
from narwhals.utils import horizontal_concat
from narwhals.utils import parse_into_exprs
from narwhals.utils import series_from_iterable


class Namespace(NamespaceProtocol):
    def __init__(self, *, api_version: str, implementation: str) -> None:
        self.__dataframeapi_version__ = api_version
        self.api_version = api_version
        self._implementation = implementation

    # --- horizontal reductions
    def sum_horizontal(self, *exprs: IntoExpr | Iterable[IntoExpr]) -> ExprT:
        return reduce(lambda x, y: x + y, parse_into_exprs(self, *exprs))

    def all_horizontal(self, *exprs: IntoExpr | Iterable[IntoExpr]) -> ExprT:
        return reduce(lambda x, y: x & y, parse_into_exprs(self, *exprs))

    def any_horizontal(self, *exprs: IntoExpr | Iterable[IntoExpr]) -> ExprT:
        return reduce(lambda x, y: x | y, parse_into_exprs(self, *exprs))

    def concat(self, items: Iterable[AnyDataFrame], *, how: str) -> AnyDataFrame:
        dfs: list[Any] = []
        kind: Any = {}
        for df in items:
            dfs.append(df.dataframe)  # type: ignore[union-attr, attr-defined]
            kind.append(type(df))
        if len(kind) > 1:
            msg = "Can only concat DataFrames or LazyFrames, not mixtures of the two"
            raise TypeError(msg)
        if how != "horizontal":
            msg = "Only horizontal concatenation is supported for now"
            raise TypeError(msg)
        if kind[0] is DataFrame:
            return DataFrame(  # type: ignore[return-value]
                horizontal_concat(dfs, implementation=self._implementation),
                api_version=self.api_version,
                implementation=self._implementation,
            )
        return LazyFrame(  # type: ignore[return-value]
            horizontal_concat(dfs, implementation=self._implementation),
            api_version=self.api_version,
            implementation=self._implementation,
        )

    def col(self, *column_names: str | Iterable[str]) -> ExprT:
        return Expr.from_column_names(
            *flatten_str(*column_names), implementation=self._implementation
        )

    def sum(self, *column_names: str) -> ExprT:
        return Expr.from_column_names(
            *column_names, implementation=self._implementation
        ).sum()

    def mean(self, *column_names: str) -> ExprT:
        return Expr.from_column_names(
            *column_names, implementation=self._implementation
        ).mean()

    def max(self, *column_names: str) -> ExprT:
        return Expr.from_column_names(
            *column_names, implementation=self._implementation
        ).max()

    def min(self, *column_names: str) -> ExprT:
        return Expr.from_column_names(
            *column_names, implementation=self._implementation
        ).min()

    def len(self) -> ExprT:
        return Expr(
            lambda df: [
                Series(
                    series_from_iterable(
                        [len(df.dataframe)],  # type: ignore[union-attr]
                        name="len",
                        index=[0],
                        implementation=self._implementation,
                    ),
                    api_version=df.api_version,  # type: ignore[union-attr]
                    implementation=self._implementation,
                ),
            ],
            depth=0,
            function_name="len",
            root_names=None,
            output_names=["len"],  # todo: check this
            implementation=self._implementation,
        )

    def _create_expr_from_callable(  # noqa: PLR0913
        self,
        func: Callable[[DataFrameT | LazyFrameProtocol], list[SeriesT]],
        *,
        depth: int,
        function_name: str | None,
        root_names: list[str] | None,
        output_names: list[str] | None,
    ) -> ExprT:
        return Expr(
            func,
            depth=depth,
            function_name=function_name,
            root_names=root_names,
            output_names=output_names,
            implementation=self._implementation,
        )

    def _create_series_from_scalar(self, value: Any, series: SeriesT) -> SeriesT:
        return Series(
            series_from_iterable(
                [value],
                name=series.series.name,  # type: ignore[attr-defined]
                index=series.series.index[0:1],  # type: ignore[attr-defined]
                implementation=self._implementation,
            ),
            api_version=self.api_version,
            implementation=self._implementation,
        )

    def _create_expr_from_series(self, series: SeriesT) -> ExprT:
        return Expr(
            lambda _df: [series],
            depth=0,
            function_name="from_series",
            root_names=None,
            output_names=None,
            implementation=self._implementation,
        )

    def all(self) -> ExprT:
        return Expr(
            lambda df: [
                Series(
                    df.dataframe.loc[:, column_name],  # type: ignore[union-attr]
                    api_version=df.api_version,  # type: ignore[union-attr]
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
