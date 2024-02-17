from __future__ import annotations
from polars_api_compat.utils import (
    register_expression_call,
    flatten_str,
    parse_into_exprs,
)

from functools import reduce
from typing import TYPE_CHECKING
from typing import Any, Iterable
from typing import Callable

import pandas as pd

from polars_api_compat.pandas.column_object import Series
from polars_api_compat.pandas.dataframe_object import LazyFrame
from polars_api_compat.spec import (
    DataFrame as DataFrameT,
    LazyFrame as LazyFrameT,
    Series as SeriesT,
    IntoExpr,
    Expr as ExprT,
    Namespace as NamespaceT,
)

if TYPE_CHECKING:
    from polars_api_compat.spec import (
        Expr as ExprT,
    )

SUPPORTED_VERSIONS = frozenset({"2023.11-beta"})


def translate(
    df: pd.DataFrame,
    api_version: str | None = None,
) -> tuple[LazyFrameT, NamespaceT]:
    df = LazyFrame(df, api_version=api_version or "2023.11-beta")
    return df, df.__lazyframe_namespace__()


class Namespace(NamespaceT):
    def __init__(self, *, api_version: str) -> None:
        self.__dataframeapi_version__ = api_version
        self.api_version = api_version

    # --- horizontal reductions
    def sum_horizontal(self, *exprs: IntoExpr | Iterable[IntoExpr]) -> ExprT:
        return reduce(lambda x, y: x + y, parse_into_exprs(self, *exprs))

    def all_horizontal(self, *exprs: IntoExpr | Iterable[IntoExpr]) -> ExprT:
        return reduce(lambda x, y: x & y, parse_into_exprs(self, *exprs))

    def any_horizontal(self, *exprs: IntoExpr | Iterable[IntoExpr]) -> ExprT:
        return reduce(lambda x, y: x | y, parse_into_exprs(self, *exprs))

    def col(self, *column_names: str | Iterable[str]) -> ExprT:
        return Expr.from_column_names(*flatten_str(*column_names))

    def sum(self, column_name: str) -> ExprT:
        return Expr.from_column_names(column_name).sum()

    def mean(self, column_name: str) -> ExprT:
        return Expr.from_column_names(column_name).mean()

    def len(self) -> ExprT:
        return Expr(
            lambda df: [
                Series(
                    pd.Series([len(df.dataframe)], name="len", index=[0]),
                    api_version=df.api_version,
                )
            ],
            depth=0,
            function_name="len",
            root_names=None,
            output_names=["len"],  # todo: check this
        )

    def _create_expr_from_callable(
        self,
        func: Callable[[DataFrameT | LazyFrameT], list[SeriesT]],
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
        )

    def _create_series_from_scalar(self, value: Any, series: SeriesT) -> SeriesT:
        return Series(
            pd.Series([value], name=series.series.name, index=series.series.index[0:1]),
            api_version=self.api_version,
        )

    def _create_expr_from_series(self, series: SeriesT) -> ExprT:
        return Expr(
            lambda df: [series],
            depth=0,
            function_name="from_series",
            root_names=None,
            output_names=None,
        )

    def all(self) -> ExprT:
        return Expr(
            lambda df: [
                Series(
                    df.dataframe.loc[:, column_name],
                    api_version=df.api_version,
                )
                for column_name in df.columns
            ],
            depth=0,
            function_name="all",
            root_names=None,
            output_names=None,
        )


class Expr(ExprT):
    def __init__(
        self,
        call: Callable[[DataFrameT | LazyFrameT], list[SeriesT]],
        depth: int | None,
        function_name: str | None,
        root_names: list[str] | None,
        output_names: list[str] | None,
    ) -> None:
        self.call = call
        self.api_version = "0.20.0"  # todo
        self.depth = depth
        self.function_name = function_name
        self.root_names = root_names
        self.depth = depth
        self.output_names = output_names

    def __repr__(self) -> str:
        return (
            f"Expr("
            f"depth={self.depth}, "
            f"function_name={self.function_name}, "
            f"root_names={self.root_names}, "
            f"output_names={self.output_names}"
        )

    @classmethod
    def from_column_names(cls: type[Expr], *column_names: str) -> ExprT:
        return cls(
            lambda df: [
                Series(
                    df.dataframe.loc[:, column_name],
                    api_version=df.api_version,
                )
                for column_name in column_names
            ],
            depth=0,
            function_name=None,
            root_names=list(column_names),
            output_names=list(column_names),
        )

    def __expr_namespace__(self) -> Namespace:
        return Namespace(api_version="2023.11-beta")

    def __eq__(self, other: Expr | Any) -> ExprT:  # type: ignore[override]
        return register_expression_call(self, "__eq__", other)

    def __ne__(self, other: Expr | Any) -> ExprT:  # type: ignore[override]
        return register_expression_call(self, "__ne__", other)

    def __ge__(self, other: Expr | Any) -> ExprT:
        return register_expression_call(self, "__ge__", other)

    def __gt__(self, other: Expr | Any) -> ExprT:
        return register_expression_call(self, "__gt__", other)

    def __le__(self, other: Expr | Any) -> ExprT:
        return register_expression_call(self, "__le__", other)

    def __lt__(self, other: Expr | Any) -> ExprT:
        return register_expression_call(self, "__lt__", other)

    def __and__(self, other: Expr | bool | Any) -> ExprT:
        return register_expression_call(self, "__and__", other)

    def __rand__(self, other: Any) -> ExprT:
        return register_expression_call(self, "__rand__", other)

    def __or__(self, other: Expr | bool | Any) -> ExprT:
        return register_expression_call(self, "__or__", other)

    def __ror__(self, other: Any) -> ExprT:
        return register_expression_call(self, "__ror__", other)

    def __add__(self, other: Expr | Any) -> ExprT:  # type: ignore[override]
        return register_expression_call(self, "__add__", other)

    def __radd__(self, other: Any) -> ExprT:
        return register_expression_call(self, "__radd__", other)

    def __sub__(self, other: Expr | Any) -> ExprT:
        return register_expression_call(self, "__sub__", other)

    def __rsub__(self, other: Any) -> ExprT:
        return register_expression_call(self, "__rsub__", other)

    def __mul__(self, other: Expr | Any) -> ExprT:
        return register_expression_call(self, "__mul__", other)

    def __rmul__(self, other: Any) -> ExprT:
        return self.__mul__(other)

    def __truediv__(self, other: Expr | Any) -> ExprT:
        return register_expression_call(self, "__truediv__", other)

    def __rtruediv__(self, other: Any) -> ExprT:
        raise NotImplementedError

    def __floordiv__(self, other: Expr | Any) -> ExprT:
        return register_expression_call(self, "__floordiv__", other)

    def __rfloordiv__(self, other: Any) -> ExprT:
        raise NotImplementedError

    def __pow__(self, other: Expr | Any) -> ExprT:
        return register_expression_call(self, "__pow__", other)

    def __rpow__(self, other: Any) -> ExprT:  # pragma: no cover
        raise NotImplementedError

    def __mod__(self, other: Expr | Any) -> ExprT:
        return register_expression_call(self, "__mod__", other)

    def __rmod__(self, other: Any) -> ExprT:  # pragma: no cover
        raise NotImplementedError

    # Unary

    def __invert__(self) -> ExprT:
        return register_expression_call(self, "__invert__")

    # Reductions

    def sum(self) -> ExprT:
        return register_expression_call(self, "sum")

    def mean(self) -> ExprT:
        return register_expression_call(self, "mean")

    # Other

    def alias(self, name: str) -> ExprT:
        # Define this one manually, so that we can
        # override `output_names`
        if self.depth is None:
            raise AssertionError("Unreachable code, please report a bug")
        return Expr(
            lambda df: [series.alias(name) for series in self.call(df)],
            depth=self.depth + 1,
            function_name=self.function_name,
            root_names=self.root_names,
            output_names=[name],
        )
