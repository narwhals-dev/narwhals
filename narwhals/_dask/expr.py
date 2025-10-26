from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Callable, cast

import pandas as pd

from narwhals._compliant import DepthTrackingExpr, LazyExpr
from narwhals._dask.expr_dt import DaskExprDateTimeNamespace
from narwhals._dask.expr_str import DaskExprStringNamespace
from narwhals._dask.utils import (
    add_row_index,
    align_series_full_broadcast,
    narwhals_to_native_dtype,
)
from narwhals._expression_parsing import evaluate_output_names_and_aliases
from narwhals._pandas_like.expr import window_kwargs_to_pandas_equivalent
from narwhals._pandas_like.utils import get_dtype_backend, native_to_narwhals_dtype
from narwhals._utils import (
    Implementation,
    generate_temporary_column_name,
    not_implemented,
)
from narwhals.exceptions import InvalidOperationError

if TYPE_CHECKING:
    from collections.abc import Sequence

    import dask.dataframe.dask_expr as dx
    from typing_extensions import Self

    from narwhals._compliant.typing import (
        AliasNames,
        EvalNames,
        EvalSeries,
        NarwhalsAggregation,
    )
    from narwhals._dask.dataframe import DaskLazyFrame
    from narwhals._dask.namespace import DaskNamespace
    from narwhals._utils import Version, _LimitedContext
    from narwhals.typing import (
        FillNullStrategy,
        IntoDType,
        ModeKeepStrategy,
        RollingInterpolationMethod,
    )


class DaskExpr(
    LazyExpr["DaskLazyFrame", "dx.Series"],  # pyright: ignore[reportInvalidTypeArguments]
    DepthTrackingExpr["DaskLazyFrame", "dx.Series"],  # pyright: ignore[reportInvalidTypeArguments]
):
    _implementation: Implementation = Implementation.DASK

    def __init__(
        self,
        call: EvalSeries[DaskLazyFrame, dx.Series],  # pyright: ignore[reportInvalidTypeForm]
        *,
        evaluate_output_names: EvalNames[DaskLazyFrame],
        alias_output_names: AliasNames | None,
        version: Version,
    ) -> None:
        self._call = call
        self._evaluate_output_names = evaluate_output_names
        self._alias_output_names = alias_output_names
        self._version = version

    def __call__(self, df: DaskLazyFrame) -> Sequence[dx.Series]:
        return self._call(df)

    def __narwhals_namespace__(self) -> DaskNamespace:  # pragma: no cover
        from narwhals._dask.namespace import DaskNamespace

        return DaskNamespace(version=self._version)

    def broadcast(self) -> Self:
        def func(df: DaskLazyFrame) -> list[dx.Series]:
            # result.loc[0][0] is a workaround for dask~<=2024.10.0/dask_expr~<=1.1.16
            #   that raised a KeyError for result[0] during collection.
            return [result.loc[0][0] for result in self(df)]

        return self.__class__(
            func,
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            version=self._version,
        )

    @classmethod
    def from_column_names(
        cls: type[Self],
        evaluate_column_names: EvalNames[DaskLazyFrame],
        /,
        *,
        context: _LimitedContext,
    ) -> Self:
        def func(df: DaskLazyFrame) -> list[dx.Series]:
            try:
                return [
                    df._native_frame[column_name]
                    for column_name in evaluate_column_names(df)
                ]
            except KeyError as e:
                if error := df._check_columns_exist(evaluate_column_names(df)):
                    raise error from e
                raise

        return cls(
            func,
            evaluate_output_names=evaluate_column_names,
            alias_output_names=None,
            version=context._version,
        )

    @classmethod
    def from_column_indices(cls, *column_indices: int, context: _LimitedContext) -> Self:
        def func(df: DaskLazyFrame) -> list[dx.Series]:
            return [df.native.iloc[:, i] for i in column_indices]

        return cls(
            func,
            evaluate_output_names=cls._eval_names_indices(column_indices),
            alias_output_names=None,
            version=context._version,
        )

    def _with_callable(
        self,
        # First argument to `call` should be `dx.Series`
        call: Callable[..., dx.Series],
        /,
        **expressifiable_args: Self,
    ) -> Self:
        def func(df: DaskLazyFrame) -> list[dx.Series]:
            native_results: list[dx.Series] = []
            native_series_list = self._call(df)
            other_native_series = {
                key: df._evaluate_single_output_expr(value)
                for key, value in expressifiable_args.items()
            }
            for native_series in native_series_list:
                result_native = call(native_series, **other_native_series)
                native_results.append(result_native)
            return native_results

        return self.__class__(
            func,
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            version=self._version,
        )

    def _with_alias_output_names(self, func: AliasNames | None, /) -> Self:
        current_alias_output_names = self._alias_output_names
        alias_output_names = (
            None
            if func is None
            else func
            if current_alias_output_names is None
            else lambda output_names: func(current_alias_output_names(output_names))
        )
        return type(self)(
            call=self._call,
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=alias_output_names,
            version=self._version,
        )

    def _with_binary(
        self,
        call: Callable[[dx.Series, Any], dx.Series],
        name: str,
        other: Any,
        *,
        reverse: bool = False,
    ) -> Self:
        result = self._with_callable(lambda expr, other: call(expr, other), other=other)
        if reverse:
            result = result.alias("literal")
        return result

    def _binary_op(self, op_name: str, other: Any) -> Self:
        return self._with_binary(
            lambda expr, other: getattr(expr, op_name)(other), op_name, other
        )

    def _reverse_binary_op(
        self, op_name: str, operator_func: Callable[..., dx.Series], other: Any
    ) -> Self:
        return self._with_binary(
            lambda expr, other: operator_func(other, expr), op_name, other, reverse=True
        )

    def __add__(self, other: Any) -> Self:
        return self._binary_op("__add__", other)

    def __sub__(self, other: Any) -> Self:
        return self._binary_op("__sub__", other)

    def __mul__(self, other: Any) -> Self:
        return self._binary_op("__mul__", other)

    def __truediv__(self, other: Any) -> Self:
        return self._binary_op("__truediv__", other)

    def __floordiv__(self, other: Any) -> Self:
        def _floordiv(
            df: DaskLazyFrame, series: dx.Series, other: dx.Series
        ) -> dx.Series:
            series, other = align_series_full_broadcast(df, series, other)
            return (series.__floordiv__(other)).where(other != 0, None)

        def func(df: DaskLazyFrame) -> list[dx.Series]:
            other_series = df._evaluate_single_output_expr(other)
            return [_floordiv(df, series, other_series) for series in self(df)]

        return self.__class__(
            func,
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            version=self._version,
        )

    def __pow__(self, other: Any) -> Self:
        return self._binary_op("__pow__", other)

    def __mod__(self, other: Any) -> Self:
        return self._binary_op("__mod__", other)

    def __eq__(self, other: object) -> Self:  # type: ignore[override]
        return self._binary_op("__eq__", other)

    def __ne__(self, other: object) -> Self:  # type: ignore[override]
        return self._binary_op("__ne__", other)

    def __ge__(self, other: Any) -> Self:
        return self._binary_op("__ge__", other)

    def __gt__(self, other: Any) -> Self:
        return self._binary_op("__gt__", other)

    def __le__(self, other: Any) -> Self:
        return self._binary_op("__le__", other)

    def __lt__(self, other: Any) -> Self:
        return self._binary_op("__lt__", other)

    def __and__(self, other: Any) -> Self:
        return self._binary_op("__and__", other)

    def __or__(self, other: Any) -> Self:
        return self._binary_op("__or__", other)

    def __rsub__(self, other: Any) -> Self:
        return self._reverse_binary_op("__rsub__", lambda a, b: a - b, other)

    def __rtruediv__(self, other: Any) -> Self:
        return self._reverse_binary_op("__rtruediv__", lambda a, b: a / b, other)

    def __rfloordiv__(self, other: Any) -> Self:
        def _rfloordiv(
            df: DaskLazyFrame, series: dx.Series, other: dx.Series
        ) -> dx.Series:
            series, other = align_series_full_broadcast(df, series, other)
            return (other.__floordiv__(series)).where(series != 0, None)

        def func(df: DaskLazyFrame) -> list[dx.Series]:
            other_native = df._evaluate_single_output_expr(other)
            return [_rfloordiv(df, series, other_native) for series in self(df)]

        return self.__class__(
            func,
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            version=self._version,
        ).alias("literal")

    def __rpow__(self, other: Any) -> Self:
        return self._reverse_binary_op("__rpow__", lambda a, b: a**b, other)

    def __rmod__(self, other: Any) -> Self:
        return self._reverse_binary_op("__rmod__", lambda a, b: a % b, other)

    def __invert__(self) -> Self:
        return self._with_callable(lambda expr: expr.__invert__())

    def mean(self) -> Self:
        return self._with_callable(lambda expr: expr.mean().to_series())

    def median(self) -> Self:
        from narwhals.exceptions import InvalidOperationError

        def func(s: dx.Series) -> dx.Series:
            dtype = native_to_narwhals_dtype(s.dtype, self._version, Implementation.DASK)
            if not dtype.is_numeric():
                msg = "`median` operation not supported for non-numeric input type."
                raise InvalidOperationError(msg)
            return s.median_approximate().to_series()

        return self._with_callable(func)

    def min(self) -> Self:
        return self._with_callable(lambda expr: expr.min().to_series())

    def max(self) -> Self:
        return self._with_callable(lambda expr: expr.max().to_series())

    def std(self, *, ddof: int) -> Self:
        return self._with_callable(lambda expr: expr.std(ddof=ddof).to_series())

    def var(self, *, ddof: int) -> Self:
        return self._with_callable(lambda expr: expr.var(ddof=ddof).to_series())

    def skew(self) -> Self:
        return self._with_callable(lambda expr: expr.skew().to_series())

    def kurtosis(self) -> Self:
        return self._with_callable(lambda expr: expr.kurtosis().to_series())

    def shift(self, n: int) -> Self:
        return self._with_callable(lambda expr: expr.shift(n))

    def cum_sum(self, *, reverse: bool) -> Self:
        if reverse:  # pragma: no cover
            # https://github.com/dask/dask/issues/11802
            msg = "`cum_sum(reverse=True)` is not supported with Dask backend"
            raise NotImplementedError(msg)

        return self._with_callable(lambda expr: expr.cumsum())

    def cum_count(self, *, reverse: bool) -> Self:
        if reverse:  # pragma: no cover
            msg = "`cum_count(reverse=True)` is not supported with Dask backend"
            raise NotImplementedError(msg)

        return self._with_callable(lambda expr: (~expr.isna()).astype(int).cumsum())

    def cum_min(self, *, reverse: bool) -> Self:
        if reverse:  # pragma: no cover
            msg = "`cum_min(reverse=True)` is not supported with Dask backend"
            raise NotImplementedError(msg)

        return self._with_callable(lambda expr: expr.cummin())

    def cum_max(self, *, reverse: bool) -> Self:
        if reverse:  # pragma: no cover
            msg = "`cum_max(reverse=True)` is not supported with Dask backend"
            raise NotImplementedError(msg)

        return self._with_callable(lambda expr: expr.cummax())

    def cum_prod(self, *, reverse: bool) -> Self:
        if reverse:  # pragma: no cover
            msg = "`cum_prod(reverse=True)` is not supported with Dask backend"
            raise NotImplementedError(msg)

        return self._with_callable(lambda expr: expr.cumprod())

    def rolling_sum(self, window_size: int, *, min_samples: int, center: bool) -> Self:
        return self._with_callable(
            lambda expr: expr.rolling(
                window=window_size, min_periods=min_samples, center=center
            ).sum()
        )

    def rolling_mean(self, window_size: int, *, min_samples: int, center: bool) -> Self:
        return self._with_callable(
            lambda expr: expr.rolling(
                window=window_size, min_periods=min_samples, center=center
            ).mean()
        )

    def rolling_var(
        self, window_size: int, *, min_samples: int, center: bool, ddof: int
    ) -> Self:
        if ddof == 1:
            return self._with_callable(
                lambda expr: expr.rolling(
                    window=window_size, min_periods=min_samples, center=center
                ).var()
            )
        msg = "Dask backend only supports `ddof=1` for `rolling_var`"
        raise NotImplementedError(msg)

    def rolling_std(
        self, window_size: int, *, min_samples: int, center: bool, ddof: int
    ) -> Self:
        if ddof == 1:
            return self._with_callable(
                lambda expr: expr.rolling(
                    window=window_size, min_periods=min_samples, center=center
                ).std()
            )
        msg = "Dask backend only supports `ddof=1` for `rolling_std`"
        raise NotImplementedError(msg)

    def sum(self) -> Self:
        return self._with_callable(lambda expr: expr.sum().to_series())

    def count(self) -> Self:
        return self._with_callable(lambda expr: expr.count().to_series())

    def round(self, decimals: int) -> Self:
        return self._with_callable(lambda expr: expr.round(decimals))

    def floor(self) -> Self:
        import dask.array as da

        return self._with_callable(da.floor)

    def ceil(self) -> Self:
        import dask.array as da

        return self._with_callable(da.ceil)

    def unique(self) -> Self:
        return self._with_callable(lambda expr: expr.unique())

    def drop_nulls(self) -> Self:
        return self._with_callable(lambda expr: expr.dropna())

    def abs(self) -> Self:
        return self._with_callable(lambda expr: expr.abs())

    def all(self) -> Self:
        return self._with_callable(
            lambda expr: expr.all(
                axis=None, skipna=True, split_every=False, out=None
            ).to_series()
        )

    def any(self) -> Self:
        return self._with_callable(
            lambda expr: expr.any(axis=0, skipna=True, split_every=False).to_series()
        )

    def fill_nan(self, value: float | None) -> Self:
        value_nullable = pd.NA if value is None else value
        value_numpy = float("nan") if value is None else value

        def func(expr: dx.Series) -> dx.Series:
            # If/when pandas exposes an API which distinguishes NaN vs null, use that.
            mask = cast("dx.Series", expr != expr)  # noqa: PLR0124
            mask = mask.fillna(False)
            fill = (
                value_nullable
                if get_dtype_backend(expr.dtype, self._implementation)
                else value_numpy
            )
            return expr.mask(mask, fill)  # pyright: ignore[reportArgumentType]

        return self._with_callable(func)

    def fill_null(
        self, value: Self | None, strategy: FillNullStrategy | None, limit: int | None
    ) -> Self:
        def func(expr: dx.Series) -> dx.Series:
            if value is not None:
                res_ser = expr.fillna(value)
            else:
                res_ser = (
                    expr.ffill(limit=limit)
                    if strategy == "forward"
                    else expr.bfill(limit=limit)
                )
            return res_ser

        return self._with_callable(func)

    def clip(self, lower_bound: Self, upper_bound: Self) -> Self:
        return self._with_callable(
            lambda expr, lower_bound, upper_bound: expr.clip(
                lower=lower_bound, upper=upper_bound
            ),
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

    def clip_lower(self, lower_bound: Self) -> Self:
        return self._with_callable(
            lambda expr, lower_bound: expr.clip(lower=lower_bound),
            lower_bound=lower_bound,
        )

    def clip_upper(self, upper_bound: Self) -> Self:
        return self._with_callable(
            lambda expr, upper_bound: expr.clip(upper=upper_bound),
            upper_bound=upper_bound,
        )

    def diff(self) -> Self:
        return self._with_callable(lambda expr: expr.diff())

    def n_unique(self) -> Self:
        return self._with_callable(lambda expr: expr.nunique(dropna=False).to_series())

    def is_null(self) -> Self:
        return self._with_callable(lambda expr: expr.isna())

    def is_nan(self) -> Self:
        def func(expr: dx.Series) -> dx.Series:
            dtype = native_to_narwhals_dtype(
                expr.dtype, self._version, self._implementation
            )
            if dtype.is_numeric():
                return expr != expr  # pyright: ignore[reportReturnType] # noqa: PLR0124
            msg = f"`.is_nan` only supported for numeric dtypes and not {dtype}, did you mean `.is_null`?"
            raise InvalidOperationError(msg)

        return self._with_callable(func)

    def len(self) -> Self:
        return self._with_callable(lambda expr: expr.size.to_series())

    def quantile(
        self, quantile: float, interpolation: RollingInterpolationMethod
    ) -> Self:
        if interpolation == "linear":

            def func(expr: dx.Series) -> dx.Series:
                if expr.npartitions > 1:
                    msg = "`Expr.quantile` is not supported for Dask backend with multiple partitions."
                    raise NotImplementedError(msg)
                return expr.quantile(
                    q=quantile, method="dask"
                ).to_series()  # pragma: no cover

            return self._with_callable(func)
        msg = "`higher`, `lower`, `midpoint`, `nearest` - interpolation methods are not supported by Dask. Please use `linear` instead."
        raise NotImplementedError(msg)

    def is_first_distinct(self) -> Self:
        def func(expr: dx.Series) -> dx.Series:
            _name = expr.name
            col_token = generate_temporary_column_name(
                n_bytes=8, columns=[_name], prefix="row_index_"
            )
            frame = add_row_index(expr.to_frame(), col_token)
            first_distinct_index = frame.groupby(_name).agg({col_token: "min"})[col_token]
            return frame[col_token].isin(first_distinct_index)

        return self._with_callable(func)

    def is_last_distinct(self) -> Self:
        def func(expr: dx.Series) -> dx.Series:
            _name = expr.name
            col_token = generate_temporary_column_name(
                n_bytes=8, columns=[_name], prefix="row_index_"
            )
            frame = add_row_index(expr.to_frame(), col_token)
            last_distinct_index = frame.groupby(_name).agg({col_token: "max"})[col_token]
            return frame[col_token].isin(last_distinct_index)

        return self._with_callable(func)

    def is_unique(self) -> Self:
        def func(expr: dx.Series) -> dx.Series:
            _name = expr.name
            return (
                expr.to_frame()
                .groupby(_name, dropna=False)
                .transform("size", meta=(_name, int))
                == 1
            )

        return self._with_callable(func)

    def is_in(self, other: Any) -> Self:
        return self._with_callable(lambda expr: expr.isin(other))

    def null_count(self) -> Self:
        return self._with_callable(lambda expr: expr.isna().sum().to_series())

    def over(self, partition_by: Sequence[str], order_by: Sequence[str]) -> Self:
        # pandas is a required dependency of dask so it's safe to import this
        from narwhals._pandas_like.group_by import PandasLikeGroupBy

        if not partition_by:
            assert order_by  # noqa: S101

            # This is something like `nw.col('a').cum_sum().order_by(key)`
            # which we can always easily support, as it doesn't require grouping.
            def func(df: DaskLazyFrame) -> Sequence[dx.Series]:
                return self(df.sort(*order_by, descending=False, nulls_last=False))
        elif not self._is_elementary():  # pragma: no cover
            msg = (
                "Only elementary expressions are supported for `.over` in dask.\n\n"
                "Please see: "
                "https://narwhals-dev.github.io/narwhals/concepts/improve_group_by_operation/"
            )
            raise NotImplementedError(msg)
        elif order_by:
            # Wrong results https://github.com/dask/dask/issues/11806.
            msg = "`over` with `order_by` is not yet supported in Dask."
            raise NotImplementedError(msg)
        else:
            leaf_node = next(self._metadata.op_nodes_reversed())
            function_name = cast("NarwhalsAggregation", leaf_node.name)
            try:
                dask_function_name = PandasLikeGroupBy._REMAP_AGGS[function_name]
            except KeyError:
                # window functions are unsupported: https://github.com/dask/dask/issues/11806
                msg = (
                    f"Unsupported function: {function_name} in `over` context.\n\n"
                    f"Supported functions are {', '.join(PandasLikeGroupBy._REMAP_AGGS)}\n"
                )
                raise NotImplementedError(msg) from None
            dask_kwargs = window_kwargs_to_pandas_equivalent(
                function_name, leaf_node.kwargs
            )

            def func(df: DaskLazyFrame) -> Sequence[dx.Series]:
                output_names, aliases = evaluate_output_names_and_aliases(self, df, [])

                with warnings.catch_warnings():
                    # https://github.com/dask/dask/issues/11804
                    warnings.filterwarnings(
                        "ignore",
                        message=".*`meta` is not specified",
                        category=UserWarning,
                    )
                    grouped = df.native.groupby(partition_by)
                    if dask_function_name == "size":
                        if len(output_names) != 1:  # pragma: no cover
                            msg = "Safety check failed, please report a bug."
                            raise AssertionError(msg)
                        res_native = grouped.transform(
                            dask_function_name, **dask_kwargs
                        ).to_frame(output_names[0])
                    else:
                        res_native = grouped[list(output_names)].transform(
                            dask_function_name, **dask_kwargs
                        )
                result_frame = df._with_native(
                    res_native.rename(columns=dict(zip(output_names, aliases)))
                ).native
                return [result_frame[name] for name in aliases]

        return self.__class__(
            func,
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            version=self._version,
        )

    def cast(self, dtype: IntoDType) -> Self:
        def func(expr: dx.Series) -> dx.Series:
            native_dtype = narwhals_to_native_dtype(dtype, self._version)
            return expr.astype(native_dtype)

        return self._with_callable(func)

    def is_finite(self) -> Self:
        import dask.array as da

        return self._with_callable(da.isfinite)

    def log(self, base: float) -> Self:
        import dask.array as da

        def _log(expr: dx.Series) -> dx.Series:
            return da.log(expr) / da.log(base)

        return self._with_callable(_log)

    def exp(self) -> Self:
        import dask.array as da

        return self._with_callable(da.exp)

    def sqrt(self) -> Self:
        import dask.array as da

        return self._with_callable(da.sqrt)

    def mode(self, *, keep: ModeKeepStrategy) -> Self:
        def func(expr: dx.Series) -> dx.Series:
            _name = expr.name
            result = expr.to_frame().mode()[_name]
            return result.head(1) if keep == "any" else result

        return self._with_callable(func)

    @property
    def str(self) -> DaskExprStringNamespace:
        return DaskExprStringNamespace(self)

    @property
    def dt(self) -> DaskExprDateTimeNamespace:
        return DaskExprDateTimeNamespace(self)

    filter = not_implemented()
    first = not_implemented()
    rank = not_implemented()
    last = not_implemented()

    # namespaces
    list: not_implemented = not_implemented()  # type: ignore[assignment]
    struct: not_implemented = not_implemented()  # type: ignore[assignment]
