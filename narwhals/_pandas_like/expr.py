from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Callable

from narwhals._pandas_like.series import PandasSeries
from narwhals._pandas_like.utils import register_expression_call

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._pandas_like.dataframe import PandasDataFrame


class PandasExpr:
    def __init__(  # noqa: PLR0913
        self,
        call: Callable[[PandasDataFrame], list[PandasSeries]],
        *,
        depth: int,
        function_name: str,
        root_names: list[str] | None,
        output_names: list[str] | None,
        implementation: str,
    ) -> None:
        self._call = call
        self._depth = depth
        self._function_name = function_name
        self._root_names = root_names
        self._depth = depth
        self._output_names = output_names
        self._implementation = implementation

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"PandasExpr("
            f"depth={self._depth}, "
            f"function_name={self._function_name}, "
            f"root_names={self._root_names}, "
            f"output_names={self._output_names}"
        )

    @classmethod
    def from_column_names(
        cls: type[Self], *column_names: str, implementation: str
    ) -> Self:
        return cls(
            lambda df: [
                PandasSeries(
                    df._dataframe.loc[:, column_name],
                    implementation=df._implementation,
                )
                for column_name in column_names
            ],
            depth=0,
            function_name="col",
            root_names=list(column_names),
            output_names=list(column_names),
            implementation=implementation,
        )

    def cast(
        self,
        dtype: Any,
    ) -> Self:
        return register_expression_call(self, "cast", dtype)

    def __eq__(self, other: PandasExpr | Any) -> Self:  # type: ignore[override]
        return register_expression_call(self, "__eq__", other)

    def __ne__(self, other: PandasExpr | Any) -> Self:  # type: ignore[override]
        return register_expression_call(self, "__ne__", other)

    def __ge__(self, other: PandasExpr | Any) -> Self:
        return register_expression_call(self, "__ge__", other)

    def __gt__(self, other: PandasExpr | Any) -> Self:
        return register_expression_call(self, "__gt__", other)

    def __le__(self, other: PandasExpr | Any) -> Self:
        return register_expression_call(self, "__le__", other)

    def __lt__(self, other: PandasExpr | Any) -> Self:
        return register_expression_call(self, "__lt__", other)

    def __and__(self, other: PandasExpr | bool | Any) -> Self:
        return register_expression_call(self, "__and__", other)

    def __rand__(self, other: Any) -> Self:
        return register_expression_call(self, "__rand__", other)

    def __or__(self, other: PandasExpr | bool | Any) -> Self:
        return register_expression_call(self, "__or__", other)

    def __ror__(self, other: Any) -> Self:
        return register_expression_call(self, "__ror__", other)

    def __add__(self, other: PandasExpr | Any) -> Self:
        return register_expression_call(self, "__add__", other)

    def __radd__(self, other: Any) -> Self:
        return register_expression_call(self, "__radd__", other)

    def __sub__(self, other: PandasExpr | Any) -> Self:
        return register_expression_call(self, "__sub__", other)

    def __rsub__(self, other: Any) -> Self:
        return register_expression_call(self, "__rsub__", other)

    def __mul__(self, other: PandasExpr | Any) -> Self:
        return register_expression_call(self, "__mul__", other)

    def __rmul__(self, other: Any) -> Self:
        return register_expression_call(self, "__rmul__", other)

    def __truediv__(self, other: PandasExpr | Any) -> Self:
        return register_expression_call(self, "__truediv__", other)

    def __rtruediv__(self, other: Any) -> Self:
        return register_expression_call(self, "__rtruediv__", other)

    def __floordiv__(self, other: PandasExpr | Any) -> Self:
        return register_expression_call(self, "__floordiv__", other)

    def __rfloordiv__(self, other: Any) -> Self:
        return register_expression_call(self, "__rfloordiv__", other)

    def __pow__(self, other: PandasExpr | Any) -> Self:
        return register_expression_call(self, "__pow__", other)

    def __rpow__(self, other: Any) -> Self:
        return register_expression_call(self, "__rpow__", other)

    def __mod__(self, other: PandasExpr | Any) -> Self:
        return register_expression_call(self, "__mod__", other)

    def __rmod__(self, other: Any) -> Self:
        return register_expression_call(self, "__rmod__", other)

    # Unary

    def __invert__(self) -> Self:
        return register_expression_call(self, "__invert__")

    # Reductions

    def sum(self) -> Self:
        return register_expression_call(self, "sum")

    def mean(self) -> Self:
        return register_expression_call(self, "mean")

    def std(self) -> Self:
        return register_expression_call(self, "std")

    def any(self) -> Self:
        return register_expression_call(self, "any")

    def all(self) -> Self:
        return register_expression_call(self, "all")

    def max(self) -> Self:
        return register_expression_call(self, "max")

    def min(self) -> Self:
        return register_expression_call(self, "min")

    # Other
    def is_between(
        self, lower_bound: Any, upper_bound: Any, closed: str = "both"
    ) -> Self:
        return register_expression_call(
            self, "is_between", lower_bound, upper_bound, closed
        )

    def is_null(self) -> Self:
        return register_expression_call(self, "is_null")

    def is_in(self, other: Any) -> Self:
        return register_expression_call(self, "is_in", other)

    def drop_nulls(self) -> Self:
        return register_expression_call(self, "drop_nulls")

    def sort(self, *, descending: bool = False) -> Self:
        return register_expression_call(self, "sort", descending=descending)

    def n_unique(self) -> Self:
        return register_expression_call(self, "n_unique")

    def unique(self) -> Self:
        return register_expression_call(self, "unique")

    def sample(
        self,
        n: int | None = None,
        fraction: float | None = None,
        *,
        with_replacement: bool = False,
    ) -> Self:
        return register_expression_call(
            self, "sample", n, fraction=fraction, with_replacement=with_replacement
        )

    def alias(self, name: str) -> Self:
        # Define this one manually, so that we can
        # override `output_names` and not increase depth
        return self.__class__(
            lambda df: [series.alias(name) for series in self._call(df)],
            depth=self._depth,
            function_name=self._function_name,
            root_names=self._root_names,
            output_names=[name],
            implementation=self._implementation,
        )

    @property
    def str(self) -> PandasExprStringNamespace:
        return PandasExprStringNamespace(self)

    @property
    def dt(self) -> PandasExprDateTimeNamespace:
        return PandasExprDateTimeNamespace(self)


class PandasExprStringNamespace:
    def __init__(self, expr: PandasExpr) -> None:
        self._expr = expr

    def ends_with(self, suffix: str) -> PandasExpr:
        # TODO make a register_expression_call for namespaces

        return PandasExpr(
            lambda df: [
                PandasSeries(
                    series._series.str.endswith(suffix),
                    implementation=df._implementation,
                )
                for series in self._expr._call(df)
            ],
            depth=self._expr._depth + 1,
            function_name=f"{self._expr._function_name}->str.ends_with",
            root_names=self._expr._root_names,
            output_names=self._expr._output_names,
            implementation=self._expr._implementation,
        )


class PandasExprDateTimeNamespace:
    def __init__(self, expr: PandasExpr) -> None:
        self._expr = expr

    def year(self) -> PandasExpr:
        # TODO make a register_expression_call for namespaces

        return PandasExpr(
            lambda df: [
                PandasSeries(
                    series._series.dt.year,
                    implementation=df._implementation,
                )
                for series in self._expr._call(df)
            ],
            depth=self._expr._depth + 1,
            function_name=f"{self._expr._function_name}->dt.year",
            root_names=self._expr._root_names,
            output_names=self._expr._output_names,
            implementation=self._expr._implementation,
        )
