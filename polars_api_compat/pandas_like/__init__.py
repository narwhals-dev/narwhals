from __future__ import annotations

from functools import reduce
from typing import Any
from typing import Callable
from typing import Iterable

from polars_api_compat.pandas_like.column_object import Series
from polars_api_compat.pandas_like.dataframe_object import LazyFrame
from polars_api_compat.spec import DataFrame as DataFrameT
from polars_api_compat.spec import Expr as ExprT
from polars_api_compat.spec import IntoExpr
from polars_api_compat.spec import LazyFrame as LazyFrameT
from polars_api_compat.spec import Namespace as NamespaceT
from polars_api_compat.spec import Series as SeriesT
from polars_api_compat.utils import flatten_str
from polars_api_compat.utils import parse_into_exprs
from polars_api_compat.utils import register_expression_call
from polars_api_compat.utils import series_from_iterable


def translate(
    df: Any,
    implementation: str,
    api_version: str,
) -> tuple[LazyFrameT, NamespaceT]:
    df = LazyFrame(
        df,
        api_version=api_version,
        implementation=implementation,
    )
    return df, df.__lazyframe_namespace__()


class Namespace(NamespaceT):
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

    def len(self) -> ExprT:
        return Expr(
            lambda df: [
                Series(
                    series_from_iterable(
                        [len(df.dataframe)],
                        name="len",
                        index=[0],
                        implementation=self._implementation,
                    ),
                    api_version=df.api_version,
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
        func: Callable[[DataFrameT | LazyFrameT], list[SeriesT]],
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
                name=series.series.name,
                index=series.series.index[0:1],
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
                    df.dataframe.loc[:, column_name],
                    api_version=df.api_version,
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


class Expr(ExprT):
    def __init__(  # noqa: PLR0913
        self,
        call: Callable[[DataFrameT | LazyFrameT], list[SeriesT]],
        *,
        depth: int | None,
        function_name: str | None,
        root_names: list[str] | None,
        output_names: list[str] | None,
        implementation: str,
    ) -> None:
        self.call = call
        self.api_version = "0.20.0"  # todo
        self._depth = depth
        self._function_name = function_name
        self._root_names = root_names
        self._depth = depth
        self._output_names = output_names
        self._implementation = implementation

    def __repr__(self) -> str:
        return (
            f"Expr("
            f"depth={self._depth}, "
            f"function_name={self._function_name}, "
            f"root_names={self._root_names}, "
            f"output_names={self._output_names}"
        )

    @classmethod
    def from_column_names(
        cls: type[Expr], *column_names: str, implementation: str
    ) -> ExprT:
        return cls(
            lambda df: [
                Series(
                    df.dataframe.loc[:, column_name],
                    api_version=df.api_version,
                    implementation=implementation,
                )
                for column_name in column_names
            ],
            depth=0,
            function_name=None,
            root_names=list(column_names),
            output_names=list(column_names),
            implementation=implementation,
        )

    def __expr_namespace__(self) -> Namespace:
        return Namespace(
            api_version="todo",
            implementation=self._implementation,  # type: ignore[attr-defined]
        )

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
        if self._depth is None:
            msg = "Unreachable code, please report a bug"
            raise AssertionError(msg)
        return Expr(
            lambda df: [series.alias(name) for series in self.call(df)],
            depth=self._depth + 1,
            function_name=self._function_name,
            root_names=self._root_names,
            output_names=[name],
            implementation=self._implementation,
        )
