from __future__ import annotations

from functools import reduce
from typing import Any
from typing import Callable
from typing import Iterable

from narwhals.pandas_like.column_object import Series
from narwhals.pandas_like.dataframe_object import DataFrame
from narwhals.pandas_like.dataframe_object import LazyFrame
from narwhals.spec import AnyDataFrame
from narwhals.spec import DataFrame as DataFrameT
from narwhals.spec import Expr as ExprT
from narwhals.spec import ExprStringNamespace as ExprStringNamespaceT
from narwhals.spec import IntoExpr
from narwhals.spec import LazyFrame as LazyFrameT
from narwhals.spec import Namespace as NamespaceT
from narwhals.spec import Series as SeriesT
from narwhals.utils import flatten_str
from narwhals.utils import horizontal_concat
from narwhals.utils import parse_into_exprs
from narwhals.utils import register_expression_call
from narwhals.utils import series_from_iterable


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
                    df.dataframe.loc[:, column_name],  # type: ignore[union-attr]
                    api_version=df.api_version,  # type: ignore[union-attr]  # type: ignore[union-attr]
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

    def max(self) -> ExprT:
        return register_expression_call(self, "max")

    def min(self) -> ExprT:
        return register_expression_call(self, "min")

    # Other
    def is_between(
        self, lower_bound: Any, upper_bound: Any, closed: str = "both"
    ) -> ExprT:
        return register_expression_call(
            self, "is_between", lower_bound, upper_bound, closed
        )

    def is_null(self) -> ExprT:
        return register_expression_call(self, "is_null")

    def is_in(self, other: Any) -> ExprT:
        return register_expression_call(self, "is_in", other)

    def drop_nulls(self) -> ExprT:
        return register_expression_call(self, "drop_nulls")

    def n_unique(self) -> ExprT:
        return register_expression_call(self, "n_unique")

    def unique(self) -> ExprT:
        return register_expression_call(self, "unique")

    def sample(self, n: int, fraction: float, *, with_replacement: bool) -> ExprT:
        return register_expression_call(self, "sample", n, fraction, with_replacement)

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

    @property
    def str(self) -> ExprStringNamespaceT:
        return ExprStringNamespace(self)


class ExprStringNamespace(ExprStringNamespaceT):
    def __init__(self, expr: ExprT) -> None:
        self._expr = expr

    def ends_with(self, suffix: str) -> ExprT:
        # TODO make a register_expression_call for namespaces
        return Expr(
            lambda df: [
                Series(
                    series.series.str.endswith(suffix),
                    api_version=df.api_version,  # type: ignore[union-attr]
                    implementation=df._implementation,  # type: ignore[union-attr]
                )
                for series in self._expr.call(df)  # type: ignore[attr-defined]
            ],
            depth=self._expr._depth + 1,  # type: ignore[attr-defined]
            function_name=self._expr._function_name,  # type: ignore[attr-defined]
            root_names=self._expr._root_names,  # type: ignore[attr-defined]
            output_names=self._expr._output_names,  # type: ignore[attr-defined]
            implementation=self._expr._implementation,  # type: ignore[attr-defined]
        )

    def strip_chars(self, characters: str = " ") -> ExprT:
        return Expr(
            lambda df: [
                Series(
                    series.series.str.strip(characters),  # type: ignore[attr-defined]
                    api_version=df.api_version,  # type: ignore[union-attr]
                    implementation=df._implementation,  # type: ignore[union-attr]
                )
                for series in self._expr.call(df)  # type: ignore[attr-defined]
            ],
            depth=self._expr._depth + 1,  # type: ignore[attr-defined]
            function_name=self._expr._function_name,  # type: ignore[attr-defined]
            root_names=self._expr._root_names,  # type: ignore[attr-defined]
            output_names=self._expr._output_names,  # type: ignore[attr-defined]
            implementation=self._expr._implementation,  # type: ignore[attr-defined]
        )
