from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Literal
from typing import Sequence

from narwhals._ibis.expr_dt import IbisExprDateTimeNamespace
from narwhals._ibis.expr_list import IbisExprListNamespace
from narwhals._ibis.expr_name import IbisExprNameNamespace
from narwhals._ibis.expr_str import IbisExprStringNamespace
from narwhals._ibis.utils import ExprKind
from narwhals._ibis.utils import maybe_evaluate
from narwhals._ibis.utils import n_ary_operation_expr_kind
from narwhals._ibis.utils import narwhals_to_native_dtype
from narwhals.dependencies import get_ibis
from narwhals.typing import CompliantExpr
from narwhals.utils import Implementation

if TYPE_CHECKING:
    import ibis.expr.types as ir
    from typing_extensions import Self

    from narwhals._ibis.dataframe import IbisLazyFrame
    from narwhals._ibis.namespace import IbisNamespace
    from narwhals.dtypes import DType
    from narwhals.utils import Version


class IbisExpr(CompliantExpr["ir.Expr"]):  # type: ignore[type-var]
    _implementation = Implementation.IBIS
    _depth = 0  # Unused, just for compatibility with CompliantExpr

    def __init__(
        self: Self,
        call: Callable[[IbisLazyFrame], Sequence[ir.Expr]],
        *,
        function_name: str,
        evaluate_output_names: Callable[[IbisLazyFrame], Sequence[str]],
        alias_output_names: Callable[[Sequence[str]], Sequence[str]] | None,
        expr_kind: ExprKind,
        backend_version: tuple[int, ...],
        version: Version,
    ) -> None:
        self._call = call
        self._function_name = function_name
        self._evaluate_output_names = evaluate_output_names
        self._alias_output_names = alias_output_names
        self._expr_kind = expr_kind
        self._backend_version = backend_version
        self._version = version

    def __call__(self: Self, df: IbisLazyFrame) -> Sequence[ir.Expr]:
        return self._call(df)

    def __narwhals_expr__(self) -> None: ...

    def __narwhals_namespace__(self) -> IbisNamespace:  # pragma: no cover
        # Unused, just for compatibility with PandasLikeExpr
        from narwhals._ibis.namespace import IbisNamespace

        return IbisNamespace(backend_version=self._backend_version, version=self._version)

    @classmethod
    def from_column_names(
        cls: type[Self],
        *column_names: str,
        backend_version: tuple[int, ...],
        version: Version,
    ) -> Self:
        def func(df: IbisLazyFrame) -> list[ir.Expr]:
            return [df._native_frame[col_name] for col_name in column_names]

        return cls(
            func,
            function_name="col",
            evaluate_output_names=lambda _df: list(column_names),
            alias_output_names=None,
            expr_kind=ExprKind.TRANSFORM,
            backend_version=backend_version,
            version=version,
        )

    @classmethod
    def from_column_indices(
        cls: type[Self],
        *column_indices: int,
        backend_version: tuple[int, ...],
        version: Version,
    ) -> Self:
        def func(df: IbisLazyFrame) -> list[ir.Expr]:
            columns = df.columns

            return [df._native_frame[columns[i]] for i in column_indices]

        return cls(
            func,
            function_name="nth",
            evaluate_output_names=lambda df: [df.columns[i] for i in column_indices],
            alias_output_names=None,
            expr_kind=ExprKind.TRANSFORM,
            backend_version=backend_version,
            version=version,
        )

    def _from_call(
        self: Self,
        call: Callable[..., ir.Expr],
        expr_name: str,
        *,
        expr_kind: ExprKind,
        **expressifiable_args: Self | Any,
    ) -> Self:
        """Create expression from callable.

        Arguments:
            call: Callable from compliant DataFrame to native Expression
            expr_name: Expression name
            expr_kind: kind of output expression
            expressifiable_args: arguments pass to expression which should be parsed
                as expressions (e.g. in `nw.col('a').is_between('b', 'c')`)
        """

        def func(df: IbisLazyFrame) -> list[ir.Expr]:
            native_series_list = self._call(df)
            other_native_series = {
                key: maybe_evaluate(df, value)
                for key, value in expressifiable_args.items()
            }
            return [
                call(native_series, **other_native_series)
                for native_series in native_series_list
            ]

        return self.__class__(
            func,
            function_name=f"{self._function_name}->{expr_name}",
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            expr_kind=expr_kind,
            backend_version=self._backend_version,
            version=self._version,
        )

    def __and__(self: Self, other: IbisExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input & other,
            "__and__",
            other=other,
            expr_kind=n_ary_operation_expr_kind(self, other),
        )

    def __or__(self: Self, other: IbisExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input | other,
            "__or__",
            other=other,
            expr_kind=n_ary_operation_expr_kind(self, other),
        )

    def __add__(self: Self, other: IbisExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input + other,
            "__add__",
            other=other,
            expr_kind=n_ary_operation_expr_kind(self, other),
        )

    def __truediv__(self: Self, other: IbisExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input / other,
            "__truediv__",
            other=other,
            expr_kind=n_ary_operation_expr_kind(self, other),
        )

    def __floordiv__(self: Self, other: IbisExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input.__floordiv__(other),
            "__floordiv__",
            other=other,
            expr_kind=n_ary_operation_expr_kind(self, other),
        )

    def __mod__(self: Self, other: IbisExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input.__mod__(other),
            "__mod__",
            other=other,
            expr_kind=n_ary_operation_expr_kind(self, other),
        )

    def __sub__(self: Self, other: IbisExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input - other,
            "__sub__",
            other=other,
            expr_kind=n_ary_operation_expr_kind(self, other),
        )

    def __mul__(self: Self, other: IbisExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input * other,
            "__mul__",
            other=other,
            expr_kind=n_ary_operation_expr_kind(self, other),
        )

    def __pow__(self: Self, other: IbisExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input**other,
            "__pow__",
            other=other,
            expr_kind=n_ary_operation_expr_kind(self, other),
        )

    def __lt__(self: Self, other: IbisExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input < other,
            "__lt__",
            other=other,
            expr_kind=n_ary_operation_expr_kind(self, other),
        )

    def __gt__(self: Self, other: IbisExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input > other,
            "__gt__",
            other=other,
            expr_kind=n_ary_operation_expr_kind(self, other),
        )

    def __le__(self: Self, other: IbisExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input <= other,
            "__le__",
            other=other,
            expr_kind=n_ary_operation_expr_kind(self, other),
        )

    def __ge__(self: Self, other: IbisExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input >= other,
            "__ge__",
            other=other,
            expr_kind=n_ary_operation_expr_kind(self, other),
        )

    def __eq__(self: Self, other: IbisExpr) -> Self:  # type: ignore[override]
        return self._from_call(
            lambda _input, other: _input == other,
            "__eq__",
            other=other,
            expr_kind=n_ary_operation_expr_kind(self, other),
        )

    def __ne__(self: Self, other: IbisExpr) -> Self:  # type: ignore[override]
        return self._from_call(
            lambda _input, other: _input != other,
            "__ne__",
            other=other,
            expr_kind=n_ary_operation_expr_kind(self, other),
        )

    def __invert__(self: Self) -> Self:
        return self._from_call(
            lambda _input: ~_input,
            "__invert__",
            expr_kind=self._expr_kind,
        )

    def alias(self: Self, name: str) -> Self:
        def alias_output_names(names: Sequence[str]) -> Sequence[str]:
            if len(names) != 1:
                msg = f"Expected function with single output, found output names: {names}"
                raise ValueError(msg)
            return [name]

        return self.__class__(
            self._call,
            function_name=self._function_name,
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=alias_output_names,
            expr_kind=self._expr_kind,
            backend_version=self._backend_version,
            version=self._version,
        )

    def abs(self: Self) -> Self:
        return self._from_call(
            lambda _input: _input.abs(),
            "abs",
            expr_kind=self._expr_kind,
        )

    def mean(self: Self) -> Self:
        return self._from_call(
            lambda _input: _input.mean(),
            "mean",
            expr_kind=ExprKind.AGGREGATION,
        )

    def skew(self: Self) -> Self:  # TODO(rwhitten577): IMPLEMENT
        def func(_input: ir.Expr) -> ir.Expr:
            count = FunctionExpression("count", _input)
            return CaseExpression(condition=(count == lit(0)), value=lit(None)).otherwise(
                CaseExpression(
                    condition=(count == lit(1)), value=lit(float("nan"))
                ).otherwise(
                    CaseExpression(condition=(count == lit(2)), value=lit(0.0)).otherwise(
                        FunctionExpression("skewness", _input)
                    )
                )
            )

        return self._from_call(func, "skew", expr_kind=ExprKind.AGGREGATION)

    def median(self: Self) -> Self:
        return self._from_call(
            lambda _input: _input.median(),
            "median",
            expr_kind=ExprKind.AGGREGATION,
        )

    def all(self: Self) -> Self:
        return self._from_call(
            lambda _input: _input.all().name(_input.get_name()),
            "all",
            expr_kind=ExprKind.AGGREGATION,
        )

    def any(self: Self) -> Self:
        return self._from_call(
            lambda _input: _input.any().name(_input.get_name()),
            "any",
            expr_kind=ExprKind.AGGREGATION,
        )

    def quantile(
        self: Self,
        quantile: float,
        interpolation: Literal["nearest", "higher", "lower", "midpoint", "linear"],
    ) -> Self:
        def func(_input: ir.Expr) -> ir.Expr:
            if interpolation == "linear":
                return _input.quantile(quantile).name(_input.get_name())
            msg = "Only linear interpolation methods are supported for Ibis quantile."
            raise NotImplementedError(msg)

        return self._from_call(
            func,
            "quantile",
            expr_kind=ExprKind.AGGREGATION,
        )

    def clip(self: Self, lower_bound: Any, upper_bound: Any) -> Self:
        return self._from_call(
            lambda _input: _input.clip(lower=lower_bound, upper=upper_bound),
            "clip",
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            expr_kind=self._expr_kind,
        )

    def sum(self: Self) -> Self:
        return self._from_call(
            lambda _input: _input.sum(),
            "sum",
            expr_kind=ExprKind.AGGREGATION,
        )

    def n_unique(self: Self) -> Self:
        return self._from_call(
            lambda _input: _input.cast("string")
            .fill_null("__NULL__")
            .nunique()
            .name(_input.get_name()),
            "n_unique",
            expr_kind=ExprKind.AGGREGATION,
        )

    def count(self: Self) -> Self:
        return self._from_call(
            lambda _input: _input.count(),
            "count",
            expr_kind=ExprKind.AGGREGATION,
        )

    def len(self: Self) -> Self:
        return self._from_call(
            lambda _input: _input.count(),
            "len",
            expr_kind=ExprKind.AGGREGATION,
        )

    def std(self: Self, ddof: int) -> Self:
        def _std(_input: ir.Expr, ddof: int) -> ir.Expr:
            if ddof not in {0, 1}:
                msg = "Ibis only supports ddof of 0 or 1"
                raise NotImplementedError(msg)

            return _input.std(how="sample" if ddof == 1 else "pop")

        return self._from_call(
            _std,
            "std",
            ddof=ddof,
            expr_kind=ExprKind.AGGREGATION,
        )

    def var(self: Self, ddof: int) -> Self:
        def _var(_input: ir.Expr, ddof: int) -> ir.Expr:
            if ddof not in {0, 1}:
                msg = "Ibis only supports ddof of 0 or 1"
                raise NotImplementedError(msg)

            return _input.var(how="sample" if ddof == 1 else "pop")

        return self._from_call(
            _var,
            "var",
            ddof=ddof,
            expr_kind=ExprKind.AGGREGATION,
        )

    def max(self: Self) -> Self:
        return self._from_call(
            lambda _input: _input.max(),
            "max",
            expr_kind=ExprKind.AGGREGATION,
        )

    def min(self: Self) -> Self:
        return self._from_call(
            lambda _input: _input.min(),
            "min",
            expr_kind=ExprKind.AGGREGATION,
        )

    def null_count(self: Self) -> Self:
        return self._from_call(
            lambda _input: _input.isnull().sum().name(_input.get_name()),
            "null_count",
            expr_kind=ExprKind.AGGREGATION,
        )

    def is_null(self: Self) -> Self:
        return self._from_call(
            lambda _input: _input.isnull(),
            "is_null",
            expr_kind=self._expr_kind,
        )

    def is_nan(self: Self) -> Self:
        return self._from_call(
            lambda _input: _input.isnan(),
            "is_nan",
            expr_kind=self._expr_kind,
        )

    def is_finite(self: Self) -> Self:
        return self._from_call(
            lambda _input: ~_input.isinf(),
            "is_finite",
            expr_kind=self._expr_kind,
        )

    def is_in(self: Self, other: Sequence[Any]) -> Self:
        return self._from_call(
            lambda _input: _input.isin(other),
            "is_in",
            expr_kind=self._expr_kind,
        )

    def is_unique(self: Self) -> Self:
        ibis = get_ibis()

        return self._from_call(
            lambda _input: _input.count().over(ibis.window(group_by=_input)) == 1,
            "is_unique",
            expr_kind=self._expr_kind,
        )

    def round(self: Self, decimals: int) -> Self:
        return self._from_call(
            lambda _input: _input.round(decimals),
            "round",
            expr_kind=self._expr_kind,
        )

    def fill_null(self: Self, value: Any, strategy: Any, limit: int | None) -> Self:
        if strategy is not None:
            msg = "`strategy` is not supported for the Ibis backend"
            raise NotImplementedError(msg)
        if limit is not None:
            msg = "`limit` is not supported for the Ibis backend"
            raise NotImplementedError(msg)

        return self._from_call(
            lambda _input: _input.fill_null(value),
            "fill_null",
            expr_kind=self._expr_kind,
        )

    def cast(self: Self, dtype: DType | type[DType]) -> Self:
        def func(_input: ir.Expr) -> ir.Expr:
            native_dtype = narwhals_to_native_dtype(dtype, self._version)
            return _input.cast(native_dtype)

        return self._from_call(
            func,
            "cast",
            expr_kind=self._expr_kind,
        )

    @property
    def str(self: Self) -> IbisExprStringNamespace:
        return IbisExprStringNamespace(self)

    @property
    def dt(self: Self) -> IbisExprDateTimeNamespace:
        return IbisExprDateTimeNamespace(self)

    @property
    def name(self: Self) -> IbisExprNameNamespace:
        return IbisExprNameNamespace(self)

    @property
    def list(self: Self) -> IbisExprListNamespace:
        return IbisExprListNamespace(self)
