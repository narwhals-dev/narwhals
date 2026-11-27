from __future__ import annotations

import warnings
from itertools import islice
from typing import TYPE_CHECKING, Any, cast

import pandas as pd

from narwhals import Expr, Implementation, generate_temporary_column_name
from narwhals.compliant import CompliantExpr, CompliantExprNameNamespace
from narwhals.exceptions import InvalidOperationError
from narwhals.utils import NO_DEFAULT
from narwhals_dask._vendored import (
    PANDAS_REMAP_AGGS,
    ExprNamespace,
    evaluate_output_names_and_aliases,
    get_dtype_backend,
    is_dtype_pyarrow,
    length_changing_hint,
    make_group_by_kwargs,
    not_implemented,
    window_kwargs_to_pandas_equivalent,
)
from narwhals_dask.expr_dt import DaskExprDateTimeNamespace
from narwhals_dask.expr_list import DaskExprListNamespace
from narwhals_dask.expr_str import DaskExprStringNamespace
from narwhals_dask.utils import (
    _schema_class,
    add_row_index,
    align_series_by_index,
    align_series_full_broadcast,
    is_arrow_backed,
    narwhals_to_native_dtype,
    native_to_narwhals_dtype,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import dask.dataframe.dask_expr as dx
    from typing_extensions import Self

    from narwhals.typing import (
        FillNullStrategy,
        IntoDType,
        ModeKeepStrategy,
        NoDefault,
        RollingInterpolationMethod,
    )
    from narwhals.utils import Version
    from narwhals_dask._vendored import (
        AliasName,
        AliasNames,
        EvalNames,
        EvalSeries,
        NarwhalsAggregation,
    )
    from narwhals_dask.dataframe import DaskLazyFrame
    from narwhals_dask.namespace import DaskNamespace


def _broadcast_arrow_literal(
    index: pd.Index, value: Any, dtype: pd.ArrowDtype
) -> pd.Series:
    if isinstance(value, float):
        import pyarrow as pa

        # A native Arrow scalar keeps a valid floating NaN distinct from null.
        value = pa.scalar(value, type=dtype.pyarrow_dtype)
    return pd.Series(value, index=index, dtype=dtype)


def floordiv_null_on_zero(dividend: dx.Series, divisor: dx.Series) -> dx.Series:
    if is_dtype_pyarrow(divisor.dtype):
        return dividend // divisor.where(divisor != 0)
    # An Arrow dividend can also have a NumPy-backed scalar divisor.
    is_nonzero = divisor != 0
    safe_divisor = divisor.where(is_nonzero, divisor.dtype.type(1))
    return (dividend // safe_divisor).where(is_nonzero, None)


def _arrow_replacement_dtype(
    new: Sequence[Any], default: DaskExpr, fallback_dtype: Any
) -> Any:
    import pyarrow as pa

    replacements = pd.Series(dtype=pd.ArrowDtype(pa.array(new).type))
    if default._metadata.is_literal and pd.api.types.is_string_dtype(fallback_dtype):
        # Dask gives an untyped None literal the same dtype as a string.
        node = next(default._metadata.op_nodes_reversed())
        if node.name == "lit" and node.kwargs["dtype"] is None:
            fallback_dtype = pd.ArrowDtype(pa.scalar(node.kwargs["value"]).type)
    if fallback_dtype == "null[pyarrow]":
        return replacements.dtype
    fallback = pd.Series(dtype=fallback_dtype)
    if get_dtype_backend(fallback_dtype) == "numpy_nullable":
        fallback = fallback.convert_dtypes(dtype_backend="pyarrow")
    if replacements.dtype == "null[pyarrow]" and fallback.dtype.kind in "biuf":
        return pd.ArrowDtype(pa.array(fallback).type)
    dtype = pd.concat([replacements, fallback]).dtype
    if not pd.api.types.is_object_dtype(dtype):
        return dtype
    if any(
        pd.api.types.is_string_dtype(dt) for dt in (replacements.dtype, fallback_dtype)
    ):
        return pd.ArrowDtype(pa.string())
    msg = "Cannot infer a common Arrow dtype. Set `return_dtype` explicitly."
    raise InvalidOperationError(msg)


def simple_aggregation(attr: str) -> Any:
    # Translate an aggregation which differs from Dask in name only.
    return lambda self, **kwargs: self._with_callable(
        lambda expr: getattr(expr, attr)(**kwargs).to_series()
    )


def simple_method(attr: str) -> Any:
    # Translate a method which differs from Dask in name only.
    return lambda self, **kwargs: self._with_callable(
        lambda expr: getattr(expr, attr)(**kwargs)
    )


def simple_binary(attr: str) -> Any:
    # Translate a binary method which differs from Dask in name only.
    return lambda self, other: self._binary_op(attr, other)


def trivial_binary_right(op: Callable[..., dx.Series]) -> Any:
    # Translate a (right) binary method which differs from Dask in name only.
    return lambda self, other: self._reverse_binary_op(op, other)


class DaskExpr(CompliantExpr["DaskLazyFrame", "dx.Series"]):  # pyright: ignore[reportInvalidTypeArguments]
    _implementation: Implementation = Implementation.DASK
    _broadcast_literal: Self | None = None

    # Methods which are simple Narwhals->Dask translations.
    # Keep this simple, resist the temptation to do anything complex or clever.
    __add__ = simple_binary("__add__")
    __sub__ = simple_binary("__sub__")
    __mul__ = simple_binary("__mul__")
    __truediv__ = simple_binary("__truediv__")
    __pow__ = simple_binary("__pow__")
    __mod__ = simple_binary("__mod__")
    __eq__ = simple_binary("__eq__")
    __ne__ = simple_binary("__ne__")
    __gt__ = simple_binary("__gt__")
    __ge__ = simple_binary("__ge__")
    __lt__ = simple_binary("__lt__")
    __le__ = simple_binary("__le__")
    __and__ = simple_binary("__and__")
    __or__ = simple_binary("__or__")
    __xor__ = simple_binary("__xor__")
    __rsub__ = trivial_binary_right(lambda x, y: x - y)
    __rtruediv__ = trivial_binary_right(lambda x, y: x / y)
    __rpow__ = trivial_binary_right(lambda x, y: x**y)
    __rmod__ = trivial_binary_right(lambda x, y: x % y)
    all = simple_aggregation("all")
    any = simple_aggregation("any")
    count = simple_aggregation("count")
    mean = simple_aggregation("mean")
    min = simple_aggregation("min")
    skew = simple_aggregation("skew")
    sum = simple_aggregation("sum")
    __invert__ = simple_method("__invert__")
    __neg__ = simple_method("__neg__")
    abs = simple_method("abs")
    diff = simple_method("diff")
    drop_nulls = simple_method("dropna")
    is_null = simple_method("isna")
    round = simple_method("round")
    unique = simple_method("unique")

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

    def __narwhals_namespace__(self) -> DaskNamespace:  # pragma: no cover
        from narwhals_dask.namespace import DaskNamespace

        return DaskNamespace(version=self._version)

    def broadcast(self) -> Self:
        # dask<=2024.10.0 raised KeyError for result[0] during collection.
        expr = self._with_callable(lambda result: result.loc[0][0])
        if self._opt_metadata is not None and self._metadata.is_literal:
            expr._broadcast_literal = self
        return expr

    def __call__(
        self, df: DaskLazyFrame, *, index: dx.Index | None = None
    ) -> Sequence[dx.Series]:
        """Preserve the dtype of an Arrow literal when expanding it into a column."""
        if self._broadcast_literal is None:
            return self._call(df)
        target_index = df.native.index if index is None else index

        def broadcast_literal(result: dx.Series) -> dx.Series:
            scalar = result.loc[0][0]
            if is_dtype_pyarrow(result.dtype):
                return target_index.map_partitions(
                    _broadcast_arrow_literal,
                    scalar,
                    result.dtype,
                    meta=result._meta.set_axis(target_index._meta),
                )
            return scalar

        return [broadcast_literal(result) for result in self._broadcast_literal(df)]

    @classmethod
    def from_column_names(
        cls: type[Self],
        evaluate_column_names: EvalNames[DaskLazyFrame],
        /,
        *,
        context: Any,
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
    def from_column_indices(cls, *column_indices: int, context: Any) -> Self:
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
        expression_args: dict[str, Self] | None = None,
        *,
        scalar_args: bool = False,
    ) -> Self:
        def func(df: DaskLazyFrame) -> list[dx.Series]:
            native_results: list[dx.Series] = []
            native_series_list = self(df)
            other_native_series = {
                key: df._evaluate_single_output_expr(value, as_scalar=scalar_args)
                for key, value in (expression_args or {}).items()
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
        other: Self,
        *,
        broadcast: bool = False,
    ) -> Self:
        def func(df: DaskLazyFrame) -> list[dx.Series]:
            # A grouped expression may have different partitions from the source frame.
            # Expand a typed literal on the index of the row-valued operand.
            right = (
                df._evaluate_single_output_expr(other)
                if other._broadcast_literal is None
                else None
            )
            index = (
                right.index
                if self._broadcast_literal is not None and right is not None
                else None
            )
            results = []
            for left in self(df, index=index):
                rhs = (
                    right
                    if right is not None
                    else df._evaluate_single_output_expr(other, index=left.index)
                )
                operands = (
                    align_series_full_broadcast(df, left, rhs)
                    if broadcast
                    else (left, rhs)
                )
                results.append(call(*operands))
            return results

        return self.__class__(
            func,
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            version=self._version,
        )

    def _binary_op(self, op_name: str, other: Any) -> Self:
        return self._with_binary(lambda expr, other: getattr(expr, op_name)(other), other)

    def _reverse_binary_op(
        self, operator_func: Callable[..., dx.Series], other: Any
    ) -> Self:
        return self._with_binary(
            lambda expr, other: operator_func(other, expr), other
        ).alias("literal")

    def __floordiv__(self, other: Any) -> Self:
        return self._with_binary(floordiv_null_on_zero, other, broadcast=True)

    def __rfloordiv__(self, other: Any) -> Self:
        return self._with_binary(
            lambda left, right: floordiv_null_on_zero(right, left), other, broadcast=True
        ).alias("literal")

    def max(self) -> Self:
        def func(expr: dx.Series) -> dx.Series:
            if pd.api.types.is_string_dtype(expr.dtype):
                # A frame reduction keeps the string dtype when a partition is all null.
                return (
                    expr.to_frame(name="value")
                    .max()["value"]
                    .to_series()
                    .astype(expr.dtype)
                )
            return expr.max().to_series()

        return self._with_callable(func)

    def kurtosis(self) -> Self:
        def func(expr: dx.Series) -> dx.Series:
            if is_dtype_pyarrow(expr.dtype) and expr.dtype.kind in "biuf":
                # pandas 2.x cannot infer kurtosis on Arrow dtypes. Dask propagates nulls.
                expr = expr.dropna().astype(expr.dtype.pyarrow_dtype.to_pandas_dtype())
            return expr.kurtosis().to_series()

        return self._with_callable(func)

    def median(self) -> Self:
        from narwhals.exceptions import InvalidOperationError

        def func(s: dx.Series) -> dx.Series:
            dtype = native_to_narwhals_dtype(s.dtype, self._version)
            if not dtype.is_numeric():
                msg = "`median` operation not supported for non-numeric input type."
                raise InvalidOperationError(msg)
            return s.median_approximate().to_series()

        return self._with_callable(func)

    def shift(self, n: int) -> Self:
        return self._with_callable(lambda expr: expr.shift(n))

    def cum_sum(self, *, reverse: bool) -> Self:
        if reverse:  # pragma: no cover
            # https://github.com/dask/dask/issues/11802
            msg = "`cum_sum(reverse=True)` is not supported with Dask backend"
            raise NotImplementedError(msg)

        def func(expr: dx.Series) -> dx.Series:
            dtype = expr.dtype
            # Arrow accumulates in the input width and raises on overflow.
            if (
                is_dtype_pyarrow(dtype)
                and dtype.kind in "iu"
                and dtype.pyarrow_dtype.bit_width < 64
            ):
                expr = expr.astype("int64[pyarrow]")
            return expr.cumsum()

        return self._with_callable(func)

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

    def floor(self) -> Self:
        import dask.array as da

        return self._with_callable(da.floor)

    def ceil(self) -> Self:
        import dask.array as da

        return self._with_callable(da.ceil)

    def fill_nan(self, value: float | None) -> Self:
        value_nullable = pd.NA if value is None else value
        value_numpy = float("nan") if value is None else value

        def func(expr: dx.Series) -> dx.Series:
            # If/when pandas exposes an API which distinguishes NaN vs null, use that.
            mask = cast("dx.Series", expr != expr)  # noqa: PLR0124
            mask = mask.fillna(False)
            fill = value_nullable if get_dtype_backend(expr.dtype) else value_numpy
            return expr.mask(mask, fill)  # pyright: ignore[reportArgumentType]

        return self._with_callable(func)

    def fill_null(
        self, value: Self | None, strategy: FillNullStrategy | None, limit: int | None
    ) -> Self:
        if value is None:
            return self._with_callable(
                lambda expr: (
                    expr.ffill(limit=limit)
                    if strategy == "forward"
                    else expr.bfill(limit=limit)
                )
            )

        def func(df: DaskLazyFrame) -> list[dx.Series]:
            fill_value = df._evaluate_single_output_expr(value)
            aligned = (align_series_full_broadcast(df, s, fill_value) for s in self(df))
            return [series.fillna(fill) for series, fill in aligned]

        return self.__class__(
            func,
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            version=self._version,
        )

    def clip(self, lower_bound: Self, upper_bound: Self) -> Self:
        return self._with_callable(
            lambda expr, lower_bound, upper_bound: expr.clip(
                lower=lower_bound, upper=upper_bound
            ),
            expression_args={"lower_bound": lower_bound, "upper_bound": upper_bound},
            scalar_args=True,
        )

    def clip_lower(self, lower_bound: Self) -> Self:
        return self._with_callable(
            lambda expr, lower_bound: expr.clip(lower=lower_bound),
            expression_args={"lower_bound": lower_bound},
            scalar_args=True,
        )

    def clip_upper(self, upper_bound: Self) -> Self:
        return self._with_callable(
            lambda expr, upper_bound: expr.clip(upper=upper_bound),
            expression_args={"upper_bound": upper_bound},
            scalar_args=True,
        )

    def n_unique(self) -> Self:
        return self._with_callable(lambda expr: expr.nunique(dropna=False).to_series())

    def _var_or_std(self, *, ddof: int, sqrt: bool) -> Self:
        def func(expr: dx.Series) -> dx.Series:
            if ddof <= 1:
                return (expr.std if sqrt else expr.var)(ddof=ddof).to_series()
            # Dask gives `inf` (and warns) when N == ddof, so rescale the population
            # variance instead and mask N <= ddof.
            n = expr.count().to_series()
            var = (expr.var(ddof=0).to_series() * n / (n - ddof)).where(n > ddof)
            return var**0.5 if sqrt else var

        return self._with_callable(func)

    def std(self, *, ddof: int) -> Self:
        return self._var_or_std(ddof=ddof, sqrt=True)

    def var(self, *, ddof: int) -> Self:
        return self._var_or_std(ddof=ddof, sqrt=False)

    def is_nan(self) -> Self:
        def func(expr: dx.Series) -> dx.Series:
            dtype = native_to_narwhals_dtype(expr.dtype, self._version)
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
            group_by_kwargs = make_group_by_kwargs(drop_null_keys=False)
            agged = cast(
                "dx.DataFrame",
                frame.groupby(_name, **group_by_kwargs).agg({col_token: "min"}),
            )
            first_distinct_index = agged[col_token]
            return frame[col_token].isin(first_distinct_index)

        return self._with_callable(func)

    def is_last_distinct(self) -> Self:
        def func(expr: dx.Series) -> dx.Series:
            _name = expr.name
            col_token = generate_temporary_column_name(
                n_bytes=8, columns=[_name], prefix="row_index_"
            )
            frame = add_row_index(expr.to_frame(), col_token)
            group_by_kwargs = make_group_by_kwargs(drop_null_keys=False)
            agged = cast(
                "dx.DataFrame",
                frame.groupby(_name, **group_by_kwargs).agg({col_token: "max"}),
            )
            last_distinct_index = agged[col_token]
            return frame[col_token].isin(last_distinct_index)

        return self._with_callable(func)

    def is_unique(self) -> Self:
        def func(expr: dx.Series) -> dx.Series:
            _name = expr.name
            group_by_kwargs = make_group_by_kwargs(drop_null_keys=False)
            return (
                expr.to_frame()
                .groupby(_name, **group_by_kwargs)
                .transform("size", meta=(_name, int))
                == 1
            )

        return self._with_callable(func)

    def is_in(self, other: Any) -> Self:
        def func(expr: dx.Series) -> dx.Series:
            result = expr.isin(other)
            if getattr(expr.dtype, "na_value", None) is pd.NA:
                dtype = "bool[pyarrow]" if is_arrow_backed(expr.dtype) else "boolean"
                return result.astype(dtype).mask(expr.isna())
            return result

        return self._with_callable(func)

    def null_count(self) -> Self:
        return self._with_callable(lambda expr: expr.isna().sum().to_series())

    def _over_without_partition_by(self, order_by: Sequence[str]) -> Self:
        # This is something like `nw.col('a').cum_sum().order_by(key)`
        # which we can always easily support, as it doesn't require grouping.
        def func(df: DaskLazyFrame) -> Sequence[dx.Series]:
            return self(df.sort(*order_by, descending=False, nulls_last=False))

        return self.__class__(
            func,
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            version=self._version,
        )

    def over(self, partition_by: Sequence[str], order_by: Sequence[str]) -> Self:
        if not partition_by:
            assert order_by  # noqa: S101
            return self._over_without_partition_by(order_by)
        # We have something like prev.leaf().over(...) (e.g. `nw.col('a').sum().over('b')`), where:
        # - `prev` must be elementwise (in the example: `nw.col('a')`)
        # - `leaf` must be a "simple" function, i.e. one that pandas supports in `transform`
        #   (in the example: `sum`)
        #
        # We first evaluate `prev` as-is, and then evaluate `leaf().over(...)`` by using `transform`
        # or other DataFrameGroupBy methods.
        meta = self._metadata
        if partition_by and (
            meta.prev is not None and not meta.prev.is_elementwise
        ):  # pragma: no cover
            msg = (
                "Only elementary expressions are supported for `.over` in dask backend "
                "when `partition_by` is specified.\n\n"
                "Please see: "
                "https://narwhals-dev.github.io/narwhals/concepts/improve_group_by_operation/"
            )
            raise NotImplementedError(msg)

        if order_by:
            # Wrong results https://github.com/dask/dask/issues/11806.
            msg = "`over` with `order_by` is not yet supported in Dask."
            raise NotImplementedError(msg)

        nodes = list(reversed(list(self._metadata.iter_nodes_reversed())))
        leaf_node = nodes[-1]
        function_name = cast("NarwhalsAggregation", leaf_node.name)
        try:
            dask_function_name = PANDAS_REMAP_AGGS[function_name]
        except KeyError:
            # window functions are unsupported: https://github.com/dask/dask/issues/11806
            msg = (
                f"Unsupported function: {function_name} in `over` context.\n\n"
                f"Supported functions are {', '.join(PANDAS_REMAP_AGGS)}\n"
            )
            raise NotImplementedError(msg) from None
        dask_kwargs = window_kwargs_to_pandas_equivalent(function_name, leaf_node.kwargs)

        def func(df: DaskLazyFrame) -> Sequence[dx.Series]:
            plx = self.__narwhals_namespace__()
            if meta.prev is not None:
                prev = cast("DaskExpr", Expr(*nodes[:-1])._to_compliant_expr(plx))
                df = df.with_columns(prev)
            _, aliases = evaluate_output_names_and_aliases(self, df, [])

            with warnings.catch_warnings():
                # https://github.com/dask/dask/issues/11804
                warnings.filterwarnings(
                    "ignore", message=".*`meta` is not specified", category=UserWarning
                )
                group_by_kwargs = make_group_by_kwargs(drop_null_keys=False)
                grouped = df.native.groupby(partition_by, **group_by_kwargs)
                if dask_function_name == "size":
                    if len(aliases) != 1:  # pragma: no cover
                        msg = "Safety check failed, please report a bug."
                        raise AssertionError(msg)
                    res_native = grouped.transform(
                        dask_function_name, **dask_kwargs
                    ).to_frame(aliases[0])
                else:
                    res_native = grouped[list(aliases)].transform(
                        dask_function_name, **dask_kwargs
                    )
            result_frame = df._with_native(res_native).native
            return [result_frame[name] for name in aliases]

        return self.__class__(
            func,
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            version=self._version,
        )

    def cast(self, dtype: IntoDType) -> Self:
        def func(expr: dx.Series) -> dx.Series:
            native_dtype = narwhals_to_native_dtype(
                dtype, self._version, use_pyarrow=is_arrow_backed(expr.dtype)
            )
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

    def sin(self) -> Self:
        import dask.array as da

        return self._with_callable(da.sin)

    def cos(self) -> Self:
        import dask.array as da

        return self._with_callable(da.cos)

    def sqrt(self) -> Self:
        import dask.array as da

        return self._with_callable(da.sqrt)

    def mode(self, *, keep: ModeKeepStrategy) -> Self:
        def func(expr: dx.Series) -> dx.Series:
            _name = expr.name
            result = expr.to_frame().mode(dropna=False)[_name]
            return result.head(1, compute=False) if keep == "any" else result

        return self._with_callable(func)

    def replace_strict(
        self,
        default: DaskExpr | NoDefault,
        old: Sequence[Any],
        new: Sequence[Any],
        *,
        return_dtype: IntoDType | None,
    ) -> Self:
        if default is NO_DEFAULT:
            msg = "`replace_strict` requires an explicit value for `default` for dask backend."
            raise ValueError(msg)

        scalar_only = self._metadata.is_scalar_like and default._metadata.is_scalar_like
        mapping = dict(zip(old, new, strict=False))
        old_ = list(old)

        def replace(
            df: DaskLazyFrame, series: dx.Series, default_series: dx.Series
        ) -> dx.Series:
            if is_dtype_pyarrow(series.dtype) or (
                scalar_only and is_dtype_pyarrow(default_series.dtype)
            ):
                if return_dtype is not None:
                    dtype = _schema_class(self._version)({"": return_dtype}).to_pandas(
                        "pyarrow"
                    )[""]
                else:
                    dtype = _arrow_replacement_dtype(new, default, default_series.dtype)
                keys = pd.Index(
                    [pd.NA if key is None else key for key in mapping], dtype=object
                )
                # pandas 2.0 needs StringDtype to cast numeric values to Arrow strings.
                value_dtype = (
                    pd.StringDtype(storage="pyarrow")
                    if pd.api.types.is_string_dtype(dtype)
                    else dtype
                )
                lookup: pd.Series = pd.Series(
                    pd.array(list(mapping.values()), dtype=value_dtype),
                    index=keys,
                    dtype=dtype,
                    name=series.name,
                )

                def replace_partition(frame: pd.DataFrame) -> pd.Series:
                    part, fallback = frame["value"], frame["fallback"]
                    # Reindex directly: pandas 2 can round nullable integers in Series.map.
                    result = lookup.reindex(part.array).set_axis(part.index)
                    if dtype == "category":
                        # Infer categories from both the replacements and the used defaults.
                        values = result.astype(object).where(part.isin(old_), fallback)
                        return values.astype(dtype)
                    if dtype != "null[pyarrow]":
                        unmatched = ~part.isin(old_)
                        result.loc[unmatched] = fallback.loc[unmatched].astype(
                            value_dtype
                        )
                    return result

                aligned = align_series_by_index(
                    df, scalar_only=scalar_only, value=series, fallback=default_series
                )
                return aligned.map_partitions(
                    replace_partition, meta=(series.name, dtype)
                )

            result = series.replace(mapping).where(series.isin(old_), default_series)
            if return_dtype is not None:
                result = result.astype(
                    narwhals_to_native_dtype(return_dtype, self._version)
                )
            return result

        def func(df: DaskLazyFrame) -> list[dx.Series]:
            (default_series,) = align_series_full_broadcast(
                df, df._evaluate_single_output_expr(default)
            )
            return [replace(df, series, default_series) for series in self(df)]

        return self.__class__(
            func,
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            version=self._version,
        )

    @staticmethod
    def _eval_names_indices(indices: Sequence[int], /) -> EvalNames[DaskLazyFrame]:
        def fn(df: DaskLazyFrame) -> Sequence[str]:
            column_names = df.columns
            return [column_names[i] for i in indices]

        return fn

    def _evaluate_aliases(self, frame: DaskLazyFrame, /) -> Sequence[str]:
        names = self._evaluate_output_names(frame)
        return alias(names) if (alias := self._alias_output_names) else names

    def _is_elementary(self) -> bool:
        """Check if expr is elementary, e.g. `nw.col('a').mean()` or `nw.len()`."""
        return len(list(islice(self._metadata.op_nodes_reversed(), 3))) <= 2

    def alias(self, name: str) -> Self:
        def fn(names: Sequence[str]) -> Sequence[str]:
            if len(names) != 1:
                msg = f"Expected function with single output, found output names: {names}"
                raise ValueError(msg)
            return [name]

        return self._with_alias_output_names(fn)

    @property
    def name(self) -> DaskExprNameNamespace:
        return DaskExprNameNamespace(self)

    @property
    def str(self) -> DaskExprStringNamespace:
        return DaskExprStringNamespace(self)

    @property
    def dt(self) -> DaskExprDateTimeNamespace:
        return DaskExprDateTimeNamespace(self)

    @property
    def list(self) -> DaskExprListNamespace:
        return DaskExprListNamespace(self)

    any_value = not_implemented()
    filter = not_implemented(
        hint=length_changing_hint(
            instead_of="lf.select(nw.col('a').filter(nw.col('b') > 0))",
            use="lf.filter(nw.col('b') > 0).select('a')",
        )
    )
    first = not_implemented()
    rank = not_implemented()
    last = not_implemented()

    ewm_mean = not_implemented()
    map_batches = not_implemented()

    # namespaces
    cat: Any = not_implemented()
    map: Any = not_implemented()
    struct: Any = not_implemented()


class DaskExprNameNamespace(
    ExprNamespace[DaskExpr], CompliantExprNameNamespace[DaskExpr]
):
    def _from_callable(self, func: AliasName | None, /) -> DaskExpr:
        output_names = self._alias_output_names(func) if func else None
        return self.compliant._with_alias_output_names(output_names)
