from __future__ import annotations

import contextlib
import operator
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Literal
from typing import Sequence
from typing import cast

from duckdb import CaseExpression
from duckdb import CoalesceOperator
from duckdb import ColumnExpression
from duckdb import FunctionExpression
from duckdb.typing import DuckDBPyType

from narwhals._compliant import LazyExpr
from narwhals._duckdb.expr_dt import DuckDBExprDateTimeNamespace
from narwhals._duckdb.expr_list import DuckDBExprListNamespace
from narwhals._duckdb.expr_name import DuckDBExprNameNamespace
from narwhals._duckdb.expr_str import DuckDBExprStringNamespace
from narwhals._duckdb.expr_struct import DuckDBExprStructNamespace
from narwhals._duckdb.utils import WindowInputs
from narwhals._duckdb.utils import generate_order_by_sql
from narwhals._duckdb.utils import generate_partition_by_sql
from narwhals._duckdb.utils import lit
from narwhals._duckdb.utils import narwhals_to_native_dtype
from narwhals._expression_parsing import ExprKind
from narwhals.utils import Implementation
from narwhals.utils import not_implemented

if TYPE_CHECKING:
    import duckdb
    from typing_extensions import Self

    from narwhals._duckdb.dataframe import DuckDBLazyFrame
    from narwhals._duckdb.namespace import DuckDBNamespace
    from narwhals._duckdb.typing import WindowFunction
    from narwhals._expression_parsing import ExprMetadata
    from narwhals.dtypes import DType
    from narwhals.utils import Version
    from narwhals.utils import _FullContext

with contextlib.suppress(ImportError):  # requires duckdb>=1.3.0
    from duckdb import SQLExpression  # type: ignore[attr-defined, unused-ignore]


class DuckDBExpr(LazyExpr["DuckDBLazyFrame", "duckdb.Expression"]):
    _implementation = Implementation.DUCKDB

    def __init__(
        self: Self,
        call: Callable[[DuckDBLazyFrame], Sequence[duckdb.Expression]],
        *,
        evaluate_output_names: Callable[[DuckDBLazyFrame], Sequence[str]],
        alias_output_names: Callable[[Sequence[str]], Sequence[str]] | None,
        backend_version: tuple[int, ...],
        version: Version,
    ) -> None:
        self._call = call
        self._evaluate_output_names = evaluate_output_names
        self._alias_output_names = alias_output_names
        self._backend_version = backend_version
        self._version = version
        self._window_function: WindowFunction | None = None
        self._metadata: ExprMetadata | None = None

    def __call__(self: Self, df: DuckDBLazyFrame) -> Sequence[duckdb.Expression]:
        return self._call(df)

    def __narwhals_expr__(self) -> None: ...

    def __narwhals_namespace__(self) -> DuckDBNamespace:  # pragma: no cover
        # Unused, just for compatibility with PandasLikeExpr
        from narwhals._duckdb.namespace import DuckDBNamespace

        return DuckDBNamespace(
            backend_version=self._backend_version, version=self._version
        )

    def _with_metadata(self, metadata: ExprMetadata) -> Self:
        expr = self.__class__(
            self._call,
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            backend_version=self._backend_version,
            version=self._version,
        )
        if func := self._window_function:
            expr = expr._with_window_function(func)
        expr._metadata = metadata
        return expr

    def broadcast(self, kind: Literal[ExprKind.AGGREGATION, ExprKind.LITERAL]) -> Self:
        if kind is ExprKind.LITERAL:
            return self
        if self._backend_version < (1, 3):
            msg = "At least version 1.3 of DuckDB is required for binary operations between aggregates and columns."
            raise NotImplementedError(msg)

        template = "{expr} over ()"

        def func(df: DuckDBLazyFrame) -> Sequence[duckdb.Expression]:
            return [SQLExpression(template.format(expr=expr)) for expr in self(df)]

        return self.__class__(
            func,
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            backend_version=self._backend_version,
            version=self._version,
        )

    @classmethod
    def from_column_names(
        cls: type[Self],
        evaluate_column_names: Callable[[DuckDBLazyFrame], Sequence[str]],
        /,
        *,
        context: _FullContext,
    ) -> Self:
        def func(df: DuckDBLazyFrame) -> list[duckdb.Expression]:
            return [ColumnExpression(col_name) for col_name in evaluate_column_names(df)]

        return cls(
            func,
            evaluate_output_names=evaluate_column_names,
            alias_output_names=None,
            backend_version=context._backend_version,
            version=context._version,
        )

    @classmethod
    def from_column_indices(
        cls: type[Self], *column_indices: int, context: _FullContext
    ) -> Self:
        def func(df: DuckDBLazyFrame) -> list[duckdb.Expression]:
            columns = df.columns

            return [ColumnExpression(columns[i]) for i in column_indices]

        return cls(
            func,
            evaluate_output_names=lambda df: [df.columns[i] for i in column_indices],
            alias_output_names=None,
            backend_version=context._backend_version,
            version=context._version,
        )

    def _from_call(
        self: Self,
        call: Callable[..., duckdb.Expression],
        **expressifiable_args: Self | Any,
    ) -> Self:
        """Create expression from callable.

        Arguments:
            call: Callable from compliant DataFrame to native Expression
            expr_name: Expression name
            expressifiable_args: arguments pass to expression which should be parsed
                as expressions (e.g. in `nw.col('a').is_between('b', 'c')`)
        """

        def func(df: DuckDBLazyFrame) -> list[duckdb.Expression]:
            native_series_list = self(df)
            other_native_series = {
                key: df._evaluate_expr(value) if self._is_expr(value) else lit(value)
                for key, value in expressifiable_args.items()
            }
            return [
                call(native_series, **other_native_series)
                for native_series in native_series_list
            ]

        return self.__class__(
            func,
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            backend_version=self._backend_version,
            version=self._version,
        )

    def _with_window_function(
        self: Self,
        window_function: WindowFunction,
    ) -> Self:
        result = self.__class__(
            self._call,
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            backend_version=self._backend_version,
            version=self._version,
        )
        result._window_function = window_function
        return result

    def __and__(self: Self, other: DuckDBExpr) -> Self:
        return self._from_call(lambda _input, other: _input & other, other=other)

    def __or__(self: Self, other: DuckDBExpr) -> Self:
        return self._from_call(lambda _input, other: _input | other, other=other)

    def __add__(self: Self, other: DuckDBExpr) -> Self:
        return self._from_call(lambda _input, other: _input + other, other=other)

    def __truediv__(self: Self, other: DuckDBExpr) -> Self:
        return self._from_call(lambda _input, other: _input / other, other=other)

    def __rtruediv__(self: Self, other: DuckDBExpr) -> Self:
        return self._from_call(
            lambda _input, other: other.__truediv__(_input), other=other
        ).alias("literal")

    def __floordiv__(self: Self, other: DuckDBExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input.__floordiv__(other), other=other
        )

    def __rfloordiv__(self: Self, other: DuckDBExpr) -> Self:
        return self._from_call(
            lambda _input, other: other.__floordiv__(_input), other=other
        ).alias("literal")

    def __mod__(self: Self, other: DuckDBExpr) -> Self:
        return self._from_call(lambda _input, other: _input.__mod__(other), other=other)

    def __rmod__(self: Self, other: DuckDBExpr) -> Self:
        return self._from_call(
            lambda _input, other: other.__mod__(_input), other=other
        ).alias("literal")

    def __sub__(self: Self, other: DuckDBExpr) -> Self:
        return self._from_call(lambda _input, other: _input - other, other=other)

    def __rsub__(self: Self, other: DuckDBExpr) -> Self:
        return self._from_call(
            lambda _input, other: other.__sub__(_input), other=other
        ).alias("literal")

    def __mul__(self: Self, other: DuckDBExpr) -> Self:
        return self._from_call(lambda _input, other: _input * other, other=other)

    def __pow__(self: Self, other: DuckDBExpr) -> Self:
        return self._from_call(lambda _input, other: _input**other, other=other)

    def __rpow__(self: Self, other: DuckDBExpr) -> Self:
        return self._from_call(
            lambda _input, other: other.__pow__(_input), other=other
        ).alias("literal")

    def __lt__(self: Self, other: DuckDBExpr) -> Self:
        return self._from_call(lambda _input, other: _input < other, other=other)

    def __gt__(self: Self, other: DuckDBExpr) -> Self:
        return self._from_call(lambda _input, other: _input > other, other=other)

    def __le__(self: Self, other: DuckDBExpr) -> Self:
        return self._from_call(lambda _input, other: _input <= other, other=other)

    def __ge__(self: Self, other: DuckDBExpr) -> Self:
        return self._from_call(lambda _input, other: _input >= other, other=other)

    def __eq__(self: Self, other: DuckDBExpr) -> Self:  # type: ignore[override]
        return self._from_call(lambda _input, other: _input == other, other=other)

    def __ne__(self: Self, other: DuckDBExpr) -> Self:  # type: ignore[override]
        return self._from_call(lambda _input, other: _input != other, other=other)

    def __invert__(self: Self) -> Self:
        invert = cast("Callable[..., duckdb.Expression]", operator.invert)
        return self._from_call(invert)

    def alias(self: Self, name: str) -> Self:
        def alias_output_names(names: Sequence[str]) -> Sequence[str]:
            if len(names) != 1:
                msg = f"Expected function with single output, found output names: {names}"
                raise ValueError(msg)
            return [name]

        return self.__class__(
            self._call,
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=alias_output_names,
            backend_version=self._backend_version,
            version=self._version,
        )

    def abs(self: Self) -> Self:
        return self._from_call(lambda _input: FunctionExpression("abs", _input))

    def mean(self: Self) -> Self:
        return self._from_call(lambda _input: FunctionExpression("mean", _input))

    def skew(self: Self) -> Self:
        def func(_input: duckdb.Expression) -> duckdb.Expression:
            count = FunctionExpression("count", _input)
            return CaseExpression(condition=(count == lit(0)), value=lit(None)).otherwise(
                CaseExpression(
                    condition=(count == lit(1)), value=lit(float("nan"))
                ).otherwise(
                    CaseExpression(condition=(count == lit(2)), value=lit(0.0)).otherwise(
                        # Adjust population skewness by correction factor to get sample skewness
                        FunctionExpression("skewness", _input)
                        * (count - lit(2))
                        / FunctionExpression("sqrt", count * (count - lit(1)))
                    )
                )
            )

        return self._from_call(func)

    def median(self: Self) -> Self:
        return self._from_call(lambda _input: FunctionExpression("median", _input))

    def all(self: Self) -> Self:
        return self._from_call(lambda _input: FunctionExpression("bool_and", _input))

    def any(self: Self) -> Self:
        return self._from_call(lambda _input: FunctionExpression("bool_or", _input))

    def quantile(
        self: Self,
        quantile: float,
        interpolation: Literal["nearest", "higher", "lower", "midpoint", "linear"],
    ) -> Self:
        def func(_input: duckdb.Expression) -> duckdb.Expression:
            if interpolation == "linear":
                return FunctionExpression("quantile_cont", _input, lit(quantile))
            msg = "Only linear interpolation methods are supported for DuckDB quantile."
            raise NotImplementedError(msg)

        return self._from_call(func)

    def clip(self: Self, lower_bound: Any, upper_bound: Any) -> Self:
        def func(
            _input: duckdb.Expression, lower_bound: Any, upper_bound: Any
        ) -> duckdb.Expression:
            return FunctionExpression(
                "greatest", FunctionExpression("least", _input, upper_bound), lower_bound
            )

        return self._from_call(func, lower_bound=lower_bound, upper_bound=upper_bound)

    def sum(self: Self) -> Self:
        return self._from_call(lambda _input: FunctionExpression("sum", _input))

    def n_unique(self: Self) -> Self:
        def func(_input: duckdb.Expression) -> duckdb.Expression:
            # https://stackoverflow.com/a/79338887/4451315
            return FunctionExpression(
                "array_unique", FunctionExpression("array_agg", _input)
            ) + FunctionExpression(
                "max",
                CaseExpression(condition=_input.isnotnull(), value=lit(0)).otherwise(
                    lit(1)
                ),
            )

        return self._from_call(func)

    def count(self: Self) -> Self:
        return self._from_call(lambda _input: FunctionExpression("count", _input))

    def len(self: Self) -> Self:
        return self._from_call(lambda _input: FunctionExpression("count"))

    def std(self: Self, ddof: int) -> Self:
        def _std(_input: duckdb.Expression) -> duckdb.Expression:
            n_samples = FunctionExpression("count", _input)
            # NOTE: Not implemented Error: Unable to transform python value of type '<class 'duckdb.duckdb.Expression'>' to DuckDB LogicalType
            return (
                FunctionExpression("stddev_pop", _input)
                * FunctionExpression("sqrt", n_samples)
                / (FunctionExpression("sqrt", (n_samples - ddof)))  # type: ignore[operator]
            )

        return self._from_call(_std)

    def var(self: Self, ddof: int) -> Self:
        def _var(_input: duckdb.Expression) -> duckdb.Expression:
            n_samples = FunctionExpression("count", _input)
            # NOTE: Not implemented Error: Unable to transform python value of type '<class 'duckdb.duckdb.Expression'>' to DuckDB LogicalType
            return FunctionExpression("var_pop", _input) * n_samples / (n_samples - ddof)  # type: ignore[operator, no-any-return]

        return self._from_call(_var)

    def max(self: Self) -> Self:
        return self._from_call(lambda _input: FunctionExpression("max", _input))

    def min(self: Self) -> Self:
        return self._from_call(lambda _input: FunctionExpression("min", _input))

    def null_count(self: Self) -> Self:
        return self._from_call(
            lambda _input: FunctionExpression("sum", _input.isnull().cast("int")),
        )

    def over(
        self: Self,
        partition_by: Sequence[str],
        order_by: Sequence[str] | None,
    ) -> Self:
        if self._backend_version < (1, 3):
            msg = "At least version 1.3 of DuckDB is required for `over` operation."
            raise NotImplementedError(msg)
        if (window_function := self._window_function) is not None:
            assert order_by is not None  # noqa: S101

            def func(df: DuckDBLazyFrame) -> list[duckdb.Expression]:
                return [
                    window_function(WindowInputs(expr, partition_by, order_by))
                    for expr in self._call(df)
                ]
        else:
            partition_by_sql = generate_partition_by_sql(*partition_by)
            template = f"{{expr}} over ({partition_by_sql})"

            def func(df: DuckDBLazyFrame) -> list[duckdb.Expression]:
                return [
                    SQLExpression(template.format(expr=expr)) for expr in self._call(df)
                ]

        return self.__class__(
            func,
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            backend_version=self._backend_version,
            version=self._version,
        )

    def is_null(self: Self) -> Self:
        return self._from_call(lambda _input: _input.isnull())

    def is_nan(self: Self) -> Self:
        return self._from_call(lambda _input: FunctionExpression("isnan", _input))

    def is_finite(self: Self) -> Self:
        return self._from_call(lambda _input: FunctionExpression("isfinite", _input))

    def is_in(self: Self, other: Sequence[Any]) -> Self:
        return self._from_call(
            lambda _input: FunctionExpression("contains", lit(other), _input)
        )

    def round(self: Self, decimals: int) -> Self:
        return self._from_call(
            lambda _input: FunctionExpression("round", _input, lit(decimals))
        )

    def shift(self, n: int) -> Self:
        def func(window_inputs: WindowInputs) -> duckdb.Expression:
            order_by_sql = generate_order_by_sql(*window_inputs.order_by, ascending=True)
            partition_by_sql = generate_partition_by_sql(*window_inputs.partition_by)
            sql = (
                f"lag({window_inputs.expr}, {n}) over ({partition_by_sql} {order_by_sql})"
            )
            return SQLExpression(sql)  # type: ignore[no-any-return, unused-ignore]

        return self._with_window_function(func)

    def is_first_distinct(self) -> Self:
        def func(window_inputs: WindowInputs) -> duckdb.Expression:
            order_by_sql = generate_order_by_sql(*window_inputs.order_by, ascending=True)
            if window_inputs.partition_by:
                partition_by_sql = (
                    generate_partition_by_sql(*window_inputs.partition_by)
                    + f", {window_inputs.expr}"
                )
            else:
                partition_by_sql = f"partition by {window_inputs.expr}"
            sql = f"row_number() over({partition_by_sql} {order_by_sql}) == 1"
            return SQLExpression(sql)  # type: ignore[no-any-return, unused-ignore]

        return self._with_window_function(func)

    def is_last_distinct(self) -> Self:
        def func(window_inputs: WindowInputs) -> duckdb.Expression:
            order_by_sql = generate_order_by_sql(*window_inputs.order_by, ascending=False)
            if window_inputs.partition_by:
                partition_by_sql = (
                    generate_partition_by_sql(*window_inputs.partition_by)
                    + f", {window_inputs.expr}"
                )
            else:
                partition_by_sql = f"partition by {window_inputs.expr}"
            sql = f"row_number() over({partition_by_sql} {order_by_sql}) == 1"
            return SQLExpression(sql)  # type: ignore[no-any-return, unused-ignore]

        return self._with_window_function(func)

    def diff(self) -> Self:
        def func(window_inputs: WindowInputs) -> duckdb.Expression:
            order_by_sql = generate_order_by_sql(*window_inputs.order_by, ascending=True)
            partition_by_sql = generate_partition_by_sql(*window_inputs.partition_by)
            sql = f"lag({window_inputs.expr}) over ({partition_by_sql} {order_by_sql})"
            return window_inputs.expr - SQLExpression(sql)  # type: ignore[no-any-return, unused-ignore]

        return self._with_window_function(func)

    def cum_sum(self, *, reverse: bool) -> Self:
        def func(window_inputs: WindowInputs) -> duckdb.Expression:
            order_by_sql = generate_order_by_sql(
                *window_inputs.order_by, ascending=not reverse
            )
            partition_by_sql = generate_partition_by_sql(*window_inputs.partition_by)
            sql = (
                f"sum ({window_inputs.expr}) over ({partition_by_sql} {order_by_sql} "
                "rows between unbounded preceding and current row)"
            )
            return SQLExpression(sql)  # type: ignore[no-any-return, unused-ignore]

        return self._with_window_function(func)

    def rolling_sum(self, window_size: int, *, min_samples: int, center: bool) -> Self:
        if center:
            half = (window_size - 1) // 2
            remainder = (window_size - 1) % 2
            start = f"{half + remainder} preceding"
            end = f"{half} following"
        else:
            start = f"{window_size - 1} preceding"
            end = "current row"

        def func(window_inputs: WindowInputs) -> duckdb.Expression:
            order_by_sql = generate_order_by_sql(*window_inputs.order_by, ascending=True)
            partition_by_sql = generate_partition_by_sql(*window_inputs.partition_by)
            window = f"({partition_by_sql} {order_by_sql} rows between {start} and {end})"
            sql = (
                f"case when count({window_inputs.expr}) over {window} >= {min_samples}"
                f"then sum({window_inputs.expr}) over {window} else null end"
            )
            return SQLExpression(sql)  # type: ignore[no-any-return, unused-ignore]

        return self._with_window_function(func)

    def fill_null(
        self: Self, value: Self | Any, strategy: Any, limit: int | None
    ) -> Self:
        if strategy is not None:
            msg = "todo"
            raise NotImplementedError(msg)

        def func(_input: duckdb.Expression, value: Any) -> duckdb.Expression:
            return CoalesceOperator(_input, value)

        return self._from_call(func, value=value)

    def cast(self: Self, dtype: DType | type[DType]) -> Self:
        def func(_input: duckdb.Expression) -> duckdb.Expression:
            native_dtype = narwhals_to_native_dtype(dtype, self._version)
            return _input.cast(DuckDBPyType(native_dtype))

        return self._from_call(func)

    @property
    def str(self: Self) -> DuckDBExprStringNamespace:
        return DuckDBExprStringNamespace(self)

    @property
    def dt(self: Self) -> DuckDBExprDateTimeNamespace:
        return DuckDBExprDateTimeNamespace(self)

    @property
    def name(self: Self) -> DuckDBExprNameNamespace:
        return DuckDBExprNameNamespace(self)

    @property
    def list(self: Self) -> DuckDBExprListNamespace:
        return DuckDBExprListNamespace(self)

    @property
    def struct(self: Self) -> DuckDBExprStructNamespace:
        return DuckDBExprStructNamespace(self)

    drop_nulls = not_implemented()
    unique = not_implemented()
    is_unique = not_implemented()
    cum_count = not_implemented()
    cum_min = not_implemented()
    cum_max = not_implemented()
    cum_prod = not_implemented()
