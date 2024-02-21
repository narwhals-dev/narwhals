from __future__ import annotations

from typing import Any
from typing import Callable

from narwhals.pandas_like.series import Series
from narwhals.spec import DataFrame as DataFrameT
from narwhals.spec import Expr as ExprT
from narwhals.spec import ExprStringNamespace as ExprStringNamespaceT
from narwhals.spec import LazyFrame as LazyFrameProtocol
from narwhals.spec import Namespace as NamespaceProtocol
from narwhals.spec import Series as SeriesT
from narwhals.utils import register_expression_call


class Expr(ExprT):
    def __init__(  # noqa: PLR0913
        self,
        call: Callable[[DataFrameT | LazyFrameProtocol], list[SeriesT]],
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

    def __expr_namespace__(self) -> NamespaceProtocol:
        from narwhals.pandas_like.namespace import Namespace

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
        # override `output_names` and not increase depth
        if self._depth is None:
            msg = "Unreachable code, please report a bug"
            raise AssertionError(msg)
        return Expr(
            lambda df: [series.alias(name) for series in self.call(df)],
            depth=self._depth,
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
