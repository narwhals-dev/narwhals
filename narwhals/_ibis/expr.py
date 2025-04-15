from __future__ import annotations

import operator
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Literal
from typing import Sequence
from typing import cast

from narwhals._compliant import LazyExpr
from narwhals._expression_parsing import ExprKind
from narwhals._ibis.expr_dt import IbisExprDateTimeNamespace
from narwhals._ibis.expr_list import IbisExprListNamespace
from narwhals._ibis.expr_str import IbisExprStringNamespace
from narwhals._ibis.expr_struct import IbisExprStructNamespace
from narwhals._ibis.utils import WindowInputs
from narwhals._ibis.utils import narwhals_to_native_dtype
from narwhals.dependencies import get_ibis
from narwhals.utils import Implementation
from narwhals.utils import not_implemented

if TYPE_CHECKING:
    import ibis.expr.types as ir
    from typing_extensions import Self

    from narwhals._compliant.typing import AliasNames
    from narwhals._compliant.typing import EvalNames
    from narwhals._compliant.typing import EvalSeries
    from narwhals._expression_parsing import ExprMetadata
    from narwhals._ibis.dataframe import IbisLazyFrame
    from narwhals._ibis.namespace import IbisNamespace
    from narwhals._ibis.typing import WindowFunction
    from narwhals.dtypes import DType
    from narwhals.typing import RankMethod
    from narwhals.typing import RollingInterpolationMethod
    from narwhals.utils import Version
    from narwhals.utils import _FullContext


class IbisExpr(LazyExpr["IbisLazyFrame", "ir.Expr"]):
    _implementation = Implementation.IBIS

    def __init__(
        self: Self,
        call: EvalSeries[IbisLazyFrame, ir.Expr],
        *,
        evaluate_output_names: EvalNames[IbisLazyFrame],
        alias_output_names: AliasNames | None,
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

    def __call__(self: Self, df: IbisLazyFrame) -> Sequence[ir.Expr]:
        return self._call(df)

    def __narwhals_expr__(self) -> None: ...

    def __narwhals_namespace__(self) -> IbisNamespace:  # pragma: no cover
        # Unused, just for compatibility with PandasLikeExpr
        from narwhals._ibis.namespace import IbisNamespace

        return IbisNamespace(backend_version=self._backend_version, version=self._version)

    def _cum_window_func(
        self,
        *,
        reverse: bool,
        func_name: Literal["sum", "max", "min", "count"],
    ) -> WindowFunction:
        def func(window_inputs: WindowInputs) -> ir.Expr:
            ibis = get_ibis()
            from ibis import _ as col

            if reverse:
                order_by_cols = [
                    ibis.desc(getattr(col, x), nulls_first=False)
                    for x in window_inputs.order_by
                ]
            else:
                order_by_cols = [
                    ibis.asc(getattr(col, x), nulls_first=True)
                    for x in window_inputs.order_by
                ]

            window = ibis.window(
                group_by=list(window_inputs.partition_by),
                order_by=order_by_cols,
                preceding=None,  # unbounded
                following=0,
            )

            return getattr(window_inputs.expr, func_name)().over(window)

        return func

    def _rolling_window_func(
        self,
        *,
        func_name: Literal["sum", "mean", "std", "var"],
        center: bool,
        window_size: int,
        min_samples: int,
        ddof: int | None = None,
    ) -> WindowFunction:
        supported_funcs = ["sum", "mean", "std", "var"]

        if center:
            preceding = window_size // 2
            following = window_size - preceding - 1
        else:
            preceding = window_size - 1
            following = 0

        def func(window_inputs: WindowInputs) -> ir.Expr:
            ibis = get_ibis()
            from ibis import _ as col

            order_by_cols = [
                ibis.asc(getattr(col, x), nulls_first=True)
                for x in window_inputs.order_by
            ]
            window = ibis.window(
                group_by=list(window_inputs.partition_by),
                order_by=order_by_cols,
                preceding=preceding,
                following=following,
            )

            expr = window_inputs.expr

            func_: ir.Expr

            if func_name in {"sum", "mean"}:
                func_ = getattr(expr, func_name)()
            elif func_name == "var" and ddof == 0:
                func_ = expr.var(how="pop")
            elif func_name in "var" and ddof == 1:
                func_ = expr.var(how="sample")
            elif func_name == "std" and ddof == 0:
                func_ = expr.std(how="pop")
            elif func_name == "std" and ddof == 1:
                func_ = expr.std(how="sample")
            elif func_name in {"var", "std"}:  # pragma: no cover
                msg = f"Only ddof=0 and ddof=1 are currently supported for rolling_{func_name}."
                raise ValueError(msg)
            else:  # pragma: no cover
                msg = f"Only the following functions are supported: {supported_funcs}.\nGot: {func_name}."
                raise ValueError(msg)

            rolling_calc = func_.over(window)

            if min_samples is not None:
                valid_count = window_inputs.expr.count().over(window)
                rolling_calc = ibis.cases(
                    (valid_count >= min_samples, rolling_calc), else_=ibis.null()
                )

            return rolling_calc

        return func

    def broadcast(self, kind: Literal[ExprKind.AGGREGATION, ExprKind.LITERAL]) -> Self:
        if kind is ExprKind.LITERAL:
            return self

        def func(df: IbisLazyFrame) -> Sequence[ir.Expr]:
            return [expr.over() for expr in self(df)]

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
        evaluate_column_names: EvalNames[IbisLazyFrame],
        /,
        *,
        context: _FullContext,
    ) -> Self:
        def func(df: IbisLazyFrame) -> list[ir.Expr]:
            return [df.native[name] for name in evaluate_column_names(df)]

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
        def func(df: IbisLazyFrame) -> list[ir.Expr]:
            columns = df.columns
            return [df.native[columns[i]] for i in column_indices]

        return cls(
            func,
            evaluate_output_names=lambda df: [df.columns[i] for i in column_indices],
            alias_output_names=None,
            backend_version=context._backend_version,
            version=context._version,
        )

    def _with_callable(
        self: Self,
        call: Callable[..., ir.Expr],
        /,
        **expressifiable_args: Self | Any,
    ) -> Self:
        """Create expression from callable.

        Arguments:
            call: Callable from compliant DataFrame to native Expression
            expr_name: Expression name
            expressifiable_args: arguments pass to expression which should be parsed
                as expressions (e.g. in `nw.col('a').is_between('b', 'c')`)
        """

        def func(df: IbisLazyFrame) -> list[ir.Expr]:
            native_series_list = self(df)
            other_native_series = {
                key: df._evaluate_expr(value) if self._is_expr(value) else value
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

    def _with_alias_output_names(self, func: AliasNames | None, /) -> Self:
        return type(self)(
            call=self._call,
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=func,
            backend_version=self._backend_version,
            version=self._version,
        )

    def _with_window_function(self, window_function: WindowFunction) -> Self:
        result = self.__class__(
            self._call,
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            backend_version=self._backend_version,
            version=self._version,
        )
        result._window_function = window_function
        return result

    def __and__(self: Self, other: IbisExpr) -> Self:
        return self._with_callable(lambda _input, other: _input & other, other=other)

    def __or__(self: Self, other: IbisExpr) -> Self:
        return self._with_callable(lambda _input, other: _input | other, other=other)

    def __add__(self: Self, other: IbisExpr) -> Self:
        return self._with_callable(lambda _input, other: _input + other, other=other)

    def __truediv__(self: Self, other: IbisExpr) -> Self:
        return self._with_callable(lambda _input, other: _input / other, other=other)

    def __rtruediv__(self: Self, other: IbisExpr) -> Self:
        return self._with_callable(
            lambda _input, other: other.__truediv__(_input), other=other
        ).alias("literal")

    def __floordiv__(self: Self, other: IbisExpr) -> Self:
        return self._with_callable(
            lambda _input, other: _input.__floordiv__(other), other=other
        )

    def __rfloordiv__(self: Self, other: IbisExpr) -> Self:
        return self._with_callable(
            lambda _input, other: other.__floordiv__(_input), other=other
        ).alias("literal")

    def __mod__(self: Self, other: IbisExpr) -> Self:
        return self._with_callable(
            lambda _input, other: _input.__mod__(other), other=other
        )

    def __rmod__(self: Self, other: IbisExpr) -> Self:
        return self._with_callable(
            lambda _input, other: other.__mod__(_input), other=other
        ).alias("literal")

    def __sub__(self: Self, other: IbisExpr) -> Self:
        return self._with_callable(lambda _input, other: _input - other, other=other)

    def __rsub__(self: Self, other: IbisExpr) -> Self:
        return self._with_callable(
            lambda _input, other: other.__sub__(_input), other=other
        ).alias("literal")

    def __mul__(self: Self, other: IbisExpr) -> Self:
        return self._with_callable(lambda _input, other: _input * other, other=other)

    def __pow__(self: Self, other: IbisExpr) -> Self:
        return self._with_callable(lambda _input, other: _input**other, other=other)

    def __rpow__(self: Self, other: IbisExpr) -> Self:
        return self._with_callable(
            lambda _input, other: other.__pow__(_input), other=other
        ).alias("literal")

    def __lt__(self: Self, other: IbisExpr) -> Self:
        return self._with_callable(lambda _input, other: _input < other, other=other)

    def __gt__(self: Self, other: IbisExpr) -> Self:
        return self._with_callable(lambda _input, other: _input > other, other=other)

    def __le__(self: Self, other: IbisExpr) -> Self:
        return self._with_callable(lambda _input, other: _input <= other, other=other)

    def __ge__(self: Self, other: IbisExpr) -> Self:
        return self._with_callable(lambda _input, other: _input >= other, other=other)

    def __eq__(self: Self, other: IbisExpr) -> Self:  # type: ignore[override]
        return self._with_callable(lambda _input, other: _input == other, other=other)

    def __ne__(self: Self, other: IbisExpr) -> Self:  # type: ignore[override]
        return self._with_callable(lambda _input, other: _input != other, other=other)

    def __invert__(self: Self) -> Self:
        invert = cast("Callable[..., ir.Expr]", operator.invert)
        return self._with_callable(invert)

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
        return self._with_callable(lambda _input: _input.abs())

    def mean(self: Self) -> Self:
        return self._with_callable(
            lambda _input: _input.mean(),
        )

    def skew(self: Self) -> Self:  # TODO(rwhitten577): IMPLEMENT
        # def func(_input: ir.Expr) -> ir.Expr:
        #     count = FunctionExpression("count", _input)
        #     return CaseExpression(condition=(count == lit(0)), value=lit(None)).otherwise(
        #         CaseExpression(
        #             condition=(count == lit(1)), value=lit(float("nan"))
        #         ).otherwise(
        #             CaseExpression(condition=(count == lit(2)), value=lit(0.0)).otherwise(
        #                 FunctionExpression("skewness", _input)
        #             )
        #         )
        #     )
        #
        # return self._from_call(func, "skew", expr_kind=ExprKind.AGGREGATION)
        raise NotImplementedError("skew is not implemented for Ibis backend")

    def median(self: Self) -> Self:
        return self._with_callable(lambda _input: _input.median())

    def all(self: Self) -> Self:
        return self._with_callable(lambda _input: _input.all().name(_input.get_name()))

    def any(self: Self) -> Self:
        return self._with_callable(lambda _input: _input.any().name(_input.get_name()))

    def quantile(
        self: Self,
        quantile: float,
        interpolation: RollingInterpolationMethod,
    ) -> Self:
        def func(_input: ir.Expr) -> ir.Expr:
            if interpolation == "linear":
                return _input.quantile(quantile).name(_input.get_name())
            msg = "Only linear interpolation methods are supported for Ibis quantile."
            raise NotImplementedError(msg)

        return self._with_callable(func)

    def clip(self: Self, lower_bound: Any, upper_bound: Any) -> Self:
        return self._with_callable(
            lambda _input: _input.clip(lower=lower_bound, upper=upper_bound)
        )

    def sum(self: Self) -> Self:
        return self._with_callable(lambda _input: _input.sum())

    def n_unique(self: Self) -> Self:
        return self._with_callable(
            lambda _input: _input.cast("string")
            .fill_null("__NULL__")
            .nunique()
            .name(_input.get_name())
        )

    def count(self: Self) -> Self:
        return self._with_callable(lambda _input: _input.count())

    def len(self: Self) -> Self:
        return self._with_callable(
            lambda _input: _input.cast("string").fill_null("__NULL__").count()
        )

    def std(self: Self, ddof: int) -> Self:
        def _std(_input: ir.Expr, ddof: int) -> ir.Expr:
            if ddof == 0:
                return _input.std(how="pop")
            elif ddof == 1:
                return _input.std(how="sample")
            else:
                n_samples = _input.count()
                std_pop = _input.std(how="pop")
                return std_pop * n_samples.sqrt() / (n_samples - ddof).sqrt()

        return self._with_callable(lambda _input: _std(_input, ddof))

    def var(self: Self, ddof: int) -> Self:
        def _var(_input: ir.Expr, ddof: int) -> ir.Expr:
            if ddof == 0:
                return _input.var(how="pop")
            elif ddof == 1:
                return _input.var(how="sample")
            else:
                n_samples = _input.count()
                var_pop = _input.var(how="pop")
                return var_pop * n_samples / (n_samples - ddof)

        return self._with_callable(lambda _input: _var(_input, ddof))

    def max(self: Self) -> Self:
        return self._with_callable(lambda _input: _input.max())

    def min(self: Self) -> Self:
        return self._with_callable(lambda _input: _input.min())

    def null_count(self: Self) -> Self:
        return self._with_callable(
            lambda _input: _input.isnull().sum().name(_input.get_name())
        )

    def over(
        self: Self,
        partition_by: Sequence[str],
        order_by: Sequence[str] | None,
    ) -> Self:
        if (window_function := self._window_function) is not None:
            assert order_by is not None  # noqa: S101

            def func(df: IbisLazyFrame) -> list[ir.Expr]:
                return [
                    window_function(WindowInputs(expr, partition_by, order_by))
                    for expr in self._call(df)
                ]
        else:

            def func(df: IbisLazyFrame) -> list[ir.Expr]:
                return [expr.over(group_by=partition_by) for expr in self._call(df)]

        return self.__class__(
            func,
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            backend_version=self._backend_version,
            version=self._version,
        )

    def is_null(self: Self) -> Self:
        return self._with_callable(lambda _input: _input.isnull())

    def is_nan(self: Self) -> Self:
        def func(_input: ir.Expr) -> ir.Expr:
            ibis = get_ibis()
            dtype = _input.type()

            if dtype.is_float64() or dtype.is_float32():
                otherwise = _input.isnan()
            else:
                otherwise = False

            return ibis.ifelse(_input.isnull(), None, otherwise)

        return self._with_callable(func)

    def is_finite(self: Self) -> Self:
        return self._with_callable(lambda _input: ~_input.isinf())

    def is_in(self: Self, other: Sequence[Any]) -> Self:
        return self._with_callable(lambda _input: _input.isin(other))

    def round(self: Self, decimals: int) -> Self:
        return self._with_callable(lambda _input: _input.round(decimals))

    def shift(self, n: int) -> Self:
        def func(window_inputs: WindowInputs) -> ir.Expr:
            return window_inputs.expr.lag(n)

        return self._with_window_function(func)

    def is_first_distinct(self) -> Self:
        def func(window_inputs: WindowInputs) -> ir.Expr:
            ibis = get_ibis()
            from ibis import _ as col

            order_by_cols = [
                ibis.asc(getattr(col, x), nulls_first=True)
                for x in window_inputs.order_by
            ]
            window = ibis.window(
                group_by=[*window_inputs.partition_by, window_inputs.expr],
                order_by=order_by_cols,
            )
            # ibis row_number starts at 0, so need to compare with 0 instead of the usual `1`
            return ibis.row_number().over(window) == 0

        return self._with_window_function(func)

    def is_last_distinct(self) -> Self:
        def func(window_inputs: WindowInputs) -> ir.Expr:
            ibis = get_ibis()
            from ibis import _ as col

            order_by_cols = [ibis.desc(getattr(col, x)) for x in window_inputs.order_by]
            window = ibis.window(
                group_by=[*window_inputs.partition_by, window_inputs.expr],
                order_by=order_by_cols,
            )
            # ibis row_number starts at 0, so need to compare with 0 instead of the usual `1`
            return ibis.row_number().over(window) == 0

        return self._with_window_function(func)

    def diff(self) -> Self:
        def func(window_inputs: WindowInputs) -> ir.Expr:
            ibis = get_ibis()
            return window_inputs.expr - window_inputs.expr.lag().over(
                ibis.window(following=0)
            )

        return self._with_window_function(func)

    def cum_sum(self, *, reverse: bool) -> Self:
        return self._with_window_function(
            self._cum_window_func(reverse=reverse, func_name="sum")
        )

    def cum_max(self, *, reverse: bool) -> Self:
        return self._with_window_function(
            self._cum_window_func(reverse=reverse, func_name="max")
        )

    def cum_min(self, *, reverse: bool) -> Self:
        return self._with_window_function(
            self._cum_window_func(reverse=reverse, func_name="min")
        )

    def cum_count(self, *, reverse: bool) -> Self:
        return self._with_window_function(
            self._cum_window_func(reverse=reverse, func_name="count")
        )

    def cum_prod(self, *, reverse: bool) -> Self:
        # https://github.com/ibis-project/ibis/issues/10542
        msg = "`cum_prod` is not supported for the Ibis backend"
        raise NotImplementedError(msg)

    def rolling_sum(self, window_size: int, *, min_samples: int, center: bool) -> Self:
        return self._with_window_function(
            self._rolling_window_func(
                func_name="sum",
                center=center,
                window_size=window_size,
                min_samples=min_samples,
            )
        )

    def rolling_mean(self, window_size: int, *, min_samples: int, center: bool) -> Self:
        return self._with_window_function(
            self._rolling_window_func(
                func_name="mean",
                center=center,
                window_size=window_size,
                min_samples=min_samples,
            )
        )

    def rolling_var(
        self, window_size: int, *, min_samples: int, center: bool, ddof: int
    ) -> Self:
        return self._with_window_function(
            self._rolling_window_func(
                func_name="var",
                center=center,
                window_size=window_size,
                min_samples=min_samples,
                ddof=ddof,
            )
        )

    def rolling_std(
        self, window_size: int, *, min_samples: int, center: bool, ddof: int
    ) -> Self:
        return self._with_window_function(
            self._rolling_window_func(
                func_name="std",
                center=center,
                window_size=window_size,
                min_samples=min_samples,
                ddof=ddof,
            )
        )

    def fill_null(
        self: Self, value: Self | Any, strategy: Any, limit: int | None
    ) -> Self:
        if strategy is not None:
            msg = "`strategy` is not supported for the Ibis backend"
            raise NotImplementedError(msg)
        if limit is not None:
            msg = "`limit` is not supported for the Ibis backend"
            raise NotImplementedError(msg)

        def _fill_null(_input: ir.Expr, value: ir.Expr) -> ir.Expr:
            return _input.fill_null(value)

        return self._with_callable(_fill_null, value=value)

    def cast(self: Self, dtype: DType | type[DType]) -> Self:
        def func(_input: ir.Expr) -> ir.Expr:
            native_dtype = narwhals_to_native_dtype(dtype, self._version)
            return _input.cast(native_dtype)

        return self._with_callable(func)

    def is_unique(self: Self) -> Self:
        ibis = get_ibis()
        return self._with_callable(
            lambda _input: _input.count().over(ibis.window(group_by=_input)) == 1
        )

    def rank(self, method: RankMethod, *, descending: bool) -> Self:
        def _rank(_input: ir.Expr) -> ir.Expr:
            ibis = get_ibis()

            order_by = _input.desc() if descending else _input.asc()
            window = ibis.window(order_by=order_by)

            if method == "dense":
                rank_ = _input.dense_rank()
            elif method == "ordinal":
                rank_ = ibis.row_number().over(window)
            else:
                rank_ = _input.rank()

            # Ibis uses 0-based ranking. Add 1 to match polars 1-based rank.
            rank_ = rank_ + 1

            # For "max" and "average", adjust using the count of rows in the partition.
            if method == "max":
                # Define a window partitioned by _input (i.e. each distinct value)
                partition = ibis.window(group_by=[_input])
                cnt = _input.count().over(partition)
                rank_ = rank_ + cnt - 1
            elif method == "average":
                partition = ibis.window(group_by=[_input])
                cnt = _input.count().over(partition)
                rank_ = rank_ + (cnt - 1) / 2.0

            return ibis.cases((_input.notnull(), rank_))

        return self._with_callable(_rank)

    @property
    def str(self: Self) -> IbisExprStringNamespace:
        return IbisExprStringNamespace(self)

    @property
    def dt(self: Self) -> IbisExprDateTimeNamespace:
        return IbisExprDateTimeNamespace(self)

    @property
    def list(self: Self) -> IbisExprListNamespace:
        return IbisExprListNamespace(self)

    @property
    def struct(self: Self) -> IbisExprStructNamespace:
        return IbisExprStructNamespace(self)

    drop_nulls = not_implemented()
    unique = not_implemented()
