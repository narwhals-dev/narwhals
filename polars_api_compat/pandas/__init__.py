from __future__ import annotations
from typing_extensions import Self
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
from polars_api_compat.pandas.dataframe_object import DataFrame
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


def convert_to_standard_compliant_dataframe(
    df: pd.DataFrame,
    api_version: str | None = None,
) -> tuple[DataFrame, NamespaceT]:
    df = DataFrame(df, api_version=api_version or "2023.11-beta")
    return df, df.__dataframe_namespace__()


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

    def col(self, *column_names: str | Iterable[str]) -> Expr:
        return Expr.from_column_names(*flatten_str(*column_names))

    def sum(self, column_name: str) -> Expr:
        return Expr.from_column_names(column_name).sum()

    def mean(self, column_name: str) -> Expr:
        return Expr.from_column_names(column_name).mean()

    def len(self) -> Expr:
        return Expr(
            lambda df: [
                Series(
                    pd.Series([len(df.dataframe)], name="len", index=[0]),
                    api_version=df.api_version,
                )
            ]
        )

    def _create_expr_from_callable(
        self, call: Callable[[DataFrameT | LazyFrameT], list[SeriesT]]
    ) -> ExprT:
        return Expr(call)

    def _create_series_from_scalar(self, value: Any, series: SeriesT) -> SeriesT:
        return Series(
            pd.Series([value], name=series.series.name, index=series.series.index[0:1]),
            api_version=self.api_version,
        )

    def _create_expr_from_series(self, series: SeriesT) -> ExprT:
        return Expr(lambda df: [series])

    def all(self) -> Expr:
        return Expr(
            lambda df: [
                Series(
                    df.dataframe.loc[:, column_name],
                    api_version=df.api_version,
                )
                for column_name in df.columns
            ],
        )


class Expr(ExprT):
    def __init__(
        self, call: Callable[[DataFrameT | LazyFrameT], list[SeriesT]]
    ) -> None:
        self.call = call
        self.api_version = "0.20.0"  # todo

    @classmethod
    def from_column_names(cls: type[Expr], *column_names: str) -> Expr:
        return cls(
            lambda df: [
                Series(
                    df.dataframe.loc[:, column_name],
                    api_version=df.api_version,
                )
                for column_name in column_names
            ],
        )

    def __expr_namespace__(self) -> Namespace:
        return Namespace(api_version="2023.11-beta")

    def __eq__(self, other: Expr | Any) -> Self:  # type: ignore[override]
        return register_expression_call(self, "__eq__", other)

    def __ne__(self, other: Expr | Any) -> Self:  # type: ignore[override]
        return register_expression_call(self, "__ne__", other)

    def __ge__(self, other: Expr | Any) -> Self:
        return register_expression_call(self, "__ge__", other)

    def __gt__(self, other: Expr | Any) -> Self:
        return register_expression_call(self, "__gt__", other)

    def __le__(self, other: Expr | Any) -> Self:
        return register_expression_call(self, "__le__", other)

    def __lt__(self, other: Expr | Any) -> Self:
        return register_expression_call(self, "__lt__", other)

    def __and__(self, other: Expr | bool | Any) -> Self:
        return register_expression_call(self, "__and__", other)

    def __rand__(self, other: Series | Any) -> Self:
        return register_expression_call(self, "__rand__", other)

    def __or__(self, other: Expr | bool | Any) -> Self:
        return register_expression_call(self, "__or__", other)

    def __ror__(self, other: Series | Any) -> Self:
        return register_expression_call(self, "__ror__", other)

    def __add__(self, other: Expr | Any) -> Self:  # type: ignore[override]
        return register_expression_call(self, "__add__", other)

    def __radd__(self, other: Series | Any) -> Self:
        return register_expression_call(self, "__radd__", other)

    def __sub__(self, other: Expr | Any) -> Self:
        return register_expression_call(self, "__sub__", other)

    def __rsub__(self, other: Series | Any) -> Self:
        return register_expression_call(self, "__rsub__", other)

    def __mul__(self, other: Expr | Any) -> Self:
        return register_expression_call(self, "__mul__", other)

    def __rmul__(self, other: Series | Any) -> Self:
        return self.__mul__(other)

    def __truediv__(self, other: Expr | Any) -> Self:
        return register_expression_call(self, "__truediv__", other)

    def __rtruediv__(self, other: Series | Any) -> Self:
        raise NotImplementedError

    def __floordiv__(self, other: Expr | Any) -> Self:
        return register_expression_call(self, "__floordiv__", other)

    def __rfloordiv__(self, other: Series | Any) -> Self:
        raise NotImplementedError

    def __pow__(self, other: Expr | Any) -> Self:
        return register_expression_call(self, "__pow__", other)

    def __rpow__(self, other: Series | Any) -> Self:  # pragma: no cover
        raise NotImplementedError

    def __mod__(self, other: Expr | Any) -> Self:
        return register_expression_call(self, "__mod__", other)

    def __rmod__(self, other: Series | Any) -> Self:  # pragma: no cover
        raise NotImplementedError

    # Unary

    def __invert__(self) -> Self:
        return register_expression_call(self, "__invert__")

    # Reductions

    def sum(self) -> ExprT:
        return register_expression_call(self, "sum")

    def mean(self) -> ExprT:
        return register_expression_call(self, "mean")

    # Other

    def alias(self, name: str) -> ExprT:
        return register_expression_call(self, "alias", name)
