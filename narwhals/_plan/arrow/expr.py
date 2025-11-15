from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, overload

import pyarrow as pa  # ignore-banned-import
import pyarrow.compute as pc  # ignore-banned-import

from narwhals._arrow.utils import narwhals_to_native_dtype
from narwhals._plan import expressions as ir
from narwhals._plan._guards import is_function_expr
from narwhals._plan.arrow import functions as fn, options as pa_options
from narwhals._plan.arrow.series import ArrowSeries as Series
from narwhals._plan.arrow.typing import ChunkedOrScalarAny, NativeScalar, StoresNativeT_co
from narwhals._plan.common import temp
from narwhals._plan.compliant.column import ExprDispatch
from narwhals._plan.compliant.expr import EagerExpr
from narwhals._plan.compliant.scalar import EagerScalar
from narwhals._plan.compliant.typing import namespace
from narwhals._plan.expressions.boolean import IsFirstDistinct, IsLastDistinct
from narwhals._plan.expressions.functions import NullCount
from narwhals._utils import Implementation, Version, _StoresNative, not_implemented
from narwhals.exceptions import InvalidOperationError, ShapeError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from typing_extensions import ParamSpec, Self, TypeAlias, TypeIs

    from narwhals._arrow.typing import ChunkedArrayAny, Incomplete
    from narwhals._plan.arrow.dataframe import ArrowDataFrame as Frame
    from narwhals._plan.arrow.namespace import ArrowNamespace
    from narwhals._plan.expressions.aggregation import (
        ArgMax,
        ArgMin,
        Count,
        First,
        Last,
        Len,
        Max,
        Mean,
        Median,
        Min,
        NUnique,
        Quantile,
        Std,
        Sum,
        Var,
    )
    from narwhals._plan.expressions.boolean import (
        All,
        IsBetween,
        IsFinite,
        IsNan,
        IsNull,
        Not,
    )
    from narwhals._plan.expressions.expr import BinaryExpr, FunctionExpr as FExpr
    from narwhals._plan.expressions.functions import (
        Abs,
        CumAgg,
        Diff,
        FillNull,
        NullCount,
        Pow,
        Rank,
        Shift,
    )
    from narwhals._plan.typing import Seq
    from narwhals._typing_compat import TypeVar
    from narwhals.typing import Into1DArray, IntoDType, PythonLiteral

    Expr: TypeAlias = "ArrowExpr"
    Scalar: TypeAlias = "ArrowScalar"

    P = ParamSpec("P")
    R_co = TypeVar(
        "R_co", bound="ChunkedOrScalarAny", covariant=True, default="ChunkedArrayAny"
    )

    class _FnNative(Protocol[P, R_co]):
        def __call__(
            self, native: ChunkedArrayAny, *args: P.args, **kwds: P.kwargs
        ) -> R_co: ...


BACKEND_VERSION = Implementation.PYARROW._backend_version()


def is_seq_column(exprs: Seq[ir.ExprIR]) -> TypeIs[Seq[ir.Column]]:
    return all(isinstance(e, ir.Column) for e in exprs)


class _ArrowDispatch(ExprDispatch["Frame", StoresNativeT_co, "ArrowNamespace"], Protocol):
    """Common to `Expr`, `Scalar` + their dependencies."""

    def __narwhals_namespace__(self) -> ArrowNamespace:
        from narwhals._plan.arrow.namespace import ArrowNamespace

        return ArrowNamespace(self.version)

    def _with_native(self, native: Any, name: str, /) -> StoresNativeT_co: ...
    def cast(self, node: ir.Cast, frame: Frame, name: str) -> StoresNativeT_co:
        data_type = narwhals_to_native_dtype(node.dtype, frame.version)
        native = node.expr.dispatch(self, frame, name).native
        return self._with_native(fn.cast(native, data_type), name)

    def pow(self, node: FExpr[Pow], frame: Frame, name: str) -> StoresNativeT_co:
        base, exponent = node.function.unwrap_input(node)
        base_ = base.dispatch(self, frame, "base").native
        exponent_ = exponent.dispatch(self, frame, "exponent").native
        return self._with_native(pc.power(base_, exponent_), name)

    def fill_null(
        self, node: FExpr[FillNull], frame: Frame, name: str
    ) -> StoresNativeT_co:
        expr, value = node.function.unwrap_input(node)
        native = expr.dispatch(self, frame, name).native
        value_ = value.dispatch(self, frame, "value").native
        return self._with_native(pc.fill_null(native, value_), name)

    def is_between(
        self, node: FExpr[IsBetween], frame: Frame, name: str
    ) -> StoresNativeT_co:
        expr, lower_bound, upper_bound = node.function.unwrap_input(node)
        native = expr.dispatch(self, frame, name).native
        lower = lower_bound.dispatch(self, frame, "lower").native
        upper = upper_bound.dispatch(self, frame, "upper").native
        result = fn.is_between(native, lower, upper, node.function.closed)
        return self._with_native(result, name)

    def _unary_function(
        self, fn_native: Callable[[Any], Any], /
    ) -> Callable[[FExpr[Any], Frame, str], StoresNativeT_co]:
        def func(node: FExpr[Any], frame: Frame, name: str) -> StoresNativeT_co:
            native = node.input[0].dispatch(self, frame, name).native
            return self._with_native(fn_native(native), name)

        return func

    def abs(self, node: FExpr[Abs], frame: Frame, name: str) -> StoresNativeT_co:
        return self._unary_function(pc.abs)(node, frame, name)

    def not_(self, node: FExpr[Not], frame: Frame, name: str) -> StoresNativeT_co:
        return self._unary_function(pc.invert)(node, frame, name)

    def all(self, node: FExpr[All], frame: Frame, name: str) -> StoresNativeT_co:
        return self._unary_function(fn.all_)(node, frame, name)

    def any(
        self, node: FExpr[ir.boolean.Any], frame: Frame, name: str
    ) -> StoresNativeT_co:
        return self._unary_function(fn.any_)(node, frame, name)

    def is_finite(
        self, node: FExpr[IsFinite], frame: Frame, name: str
    ) -> StoresNativeT_co:
        return self._unary_function(fn.is_finite)(node, frame, name)

    def is_nan(self, node: FExpr[IsNan], frame: Frame, name: str) -> StoresNativeT_co:
        return self._unary_function(fn.is_nan)(node, frame, name)

    def is_null(self, node: FExpr[IsNull], frame: Frame, name: str) -> StoresNativeT_co:
        return self._unary_function(fn.is_null)(node, frame, name)

    def binary_expr(self, node: BinaryExpr, frame: Frame, name: str) -> StoresNativeT_co:
        lhs, rhs = (
            node.left.dispatch(self, frame, name),
            node.right.dispatch(self, frame, name),
        )
        result = fn.binary(lhs.native, node.op.__class__, rhs.native)
        return self._with_native(result, name)

    def ternary_expr(
        self, node: ir.TernaryExpr, frame: Frame, name: str
    ) -> StoresNativeT_co:
        when = node.predicate.dispatch(self, frame, name)
        then = node.truthy.dispatch(self, frame, name)
        otherwise = node.falsy.dispatch(self, frame, name)
        result = pc.if_else(when.native, then.native, otherwise.native)
        return self._with_native(result, name)

    exp = not_implemented()  # type: ignore[misc]
    log = not_implemented()  # type: ignore[misc]
    sqrt = not_implemented()  # type: ignore[misc]
    round = not_implemented()  # type: ignore[misc]
    clip = not_implemented()  # type: ignore[misc]
    drop_nulls = not_implemented()  # type: ignore[misc]
    replace_strict = not_implemented()  # type: ignore[misc]
    is_in_seq = not_implemented()  # type: ignore[misc]
    is_in_expr = not_implemented()  # type: ignore[misc]
    is_in_series = not_implemented()  # type: ignore[misc]


class ArrowExpr(  # type: ignore[misc]
    _ArrowDispatch["ArrowExpr | ArrowScalar"],
    _StoresNative["ChunkedArrayAny"],
    EagerExpr["Frame", Series],
):
    _evaluated: Series
    _version: Version

    @property
    def name(self) -> str:
        return self._evaluated.name

    @classmethod
    def from_series(cls, series: Series, /) -> Self:
        obj = cls.__new__(cls)
        obj._evaluated = series
        obj._version = series.version
        return obj

    @classmethod
    def from_native(
        cls, native: ChunkedArrayAny, name: str = "", /, version: Version = Version.MAIN
    ) -> Self:
        return cls.from_series(Series.from_native(native, name, version=version))

    @overload
    def _with_native(self, result: ChunkedArrayAny, name: str, /) -> Self: ...
    @overload
    def _with_native(self, result: NativeScalar, name: str, /) -> Scalar: ...
    @overload
    def _with_native(self, result: ChunkedOrScalarAny, name: str, /) -> Scalar | Self: ...
    def _with_native(self, result: ChunkedOrScalarAny, name: str, /) -> Scalar | Self:
        if isinstance(result, pa.Scalar):
            return ArrowScalar.from_native(result, name, version=self.version)
        return self.from_native(result, name or self.name, self.version)

    # NOTE: I'm not sure what I meant by
    # > "isn't natively supported on `ChunkedArray`"
    # Was that supposed to say "is only supported on `ChunkedArray`"?
    def _dispatch_expr(self, node: ir.ExprIR, frame: Frame, name: str) -> Series:
        """Use instead of `_dispatch` *iff* an operation isn't natively supported on `ChunkedArray`.

        There is no need to broadcast, as they may have a cheaper impl elsewhere (`CompliantScalar` or `ArrowScalar`).

        Mainly for the benefit of a type checker, but the equivalent `ArrowScalar._dispatch_expr` will raise if
        the assumption fails.
        """
        return node.dispatch(self, frame, name).to_series()

    @overload
    def _function_expr(
        self, fn_native: _FnNative[P, ChunkedArrayAny], *args: P.args, **kwds: P.kwargs
    ) -> Callable[[FExpr[Any], Frame, str], Self]: ...
    @overload
    def _function_expr(
        self, fn_native: _FnNative[P, NativeScalar], *args: P.args, **kwds: P.kwargs
    ) -> Callable[[FExpr[Any], Frame, str], Scalar]: ...

    def _function_expr(
        self, fn_native: _FnNative[P, R_co], *args: P.args, **kwds: P.kwargs
    ) -> Callable[[FExpr[Any], Frame, str], Scalar | Self]:
        """Generalized `FunctionExpr` dispatcher."""

        def func(node: FExpr[Any], frame: Frame, name: str, /) -> Scalar | Self:
            native = self._dispatch_expr(node.input[0], frame, name).native
            return self._with_native(fn_native(native, *args, **kwds), name)

        return func  # type: ignore[return-value]

    @property
    def native(self) -> ChunkedArrayAny:
        return self._evaluated.native

    def to_series(self) -> Series:
        return self._evaluated

    def broadcast(self, length: int, /) -> Series:
        if (actual_len := len(self)) != length:
            msg = f"Expected object of length {length}, got {actual_len}."
            raise ShapeError(msg)
        return self._evaluated

    def __len__(self) -> int:
        return len(self._evaluated)

    def sort(self, node: ir.Sort, frame: Frame, name: str) -> Expr:
        series = self._dispatch_expr(node.expr, frame, name)
        opts = node.options
        result = series.sort(descending=opts.descending, nulls_last=opts.nulls_last)
        return self.from_series(result)

    def sort_by(self, node: ir.SortBy, frame: Frame, name: str) -> Expr:
        if is_seq_column(node.by):
            # fastpath, roughly the same as `DataFrame.sort`, but only taking indices
            # of a single column
            keys: Sequence[str] = tuple(e.name for e in node.by)
            df = frame
        else:
            it_names = temp.column_names(frame)
            by = (self._dispatch_expr(e, frame, nm) for e, nm in zip(node.by, it_names))
            df = namespace(self)._concat_horizontal(by)
            keys = df.columns
        indices = pc.sort_indices(df.native, options=node.options.to_arrow(keys))
        series = self._dispatch_expr(node.expr, frame, name)
        return self.from_series(series.gather(indices))

    def filter(self, node: ir.Filter, frame: Frame, name: str) -> Expr:
        return self._with_native(
            self._dispatch_expr(node.expr, frame, name).native.filter(
                self._dispatch_expr(node.by, frame, name).native
            ),
            name,
        )

    def first(self, node: First, frame: Frame, name: str) -> Scalar:
        prev = self._dispatch_expr(node.expr, frame, name)
        native = prev.native
        result = native[0] if len(prev) else fn.lit(None, native.type)
        return self._with_native(result, name)

    def last(self, node: Last, frame: Frame, name: str) -> Scalar:
        prev = self._dispatch_expr(node.expr, frame, name)
        native = prev.native
        result = native[len_ - 1] if (len_ := len(prev)) else fn.lit(None, native.type)
        return self._with_native(result, name)

    def arg_min(self, node: ArgMin, frame: Frame, name: str) -> Scalar:
        native = self._dispatch_expr(node.expr, frame, name).native
        result = pc.index(native, fn.min_(native))
        return self._with_native(result, name)

    def arg_max(self, node: ArgMax, frame: Frame, name: str) -> Scalar:
        native = self._dispatch_expr(node.expr, frame, name).native
        result: NativeScalar = pc.index(native, fn.max_(native))
        return self._with_native(result, name)

    def sum(self, node: Sum, frame: Frame, name: str) -> Scalar:
        result = fn.sum_(self._dispatch_expr(node.expr, frame, name).native)
        return self._with_native(result, name)

    def n_unique(self, node: NUnique, frame: Frame, name: str) -> Scalar:
        result = fn.n_unique(self._dispatch_expr(node.expr, frame, name).native)
        return self._with_native(result, name)

    def std(self, node: Std, frame: Frame, name: str) -> Scalar:
        result = fn.std(
            self._dispatch_expr(node.expr, frame, name).native, ddof=node.ddof
        )
        return self._with_native(result, name)

    def var(self, node: Var, frame: Frame, name: str) -> Scalar:
        result = fn.var(
            self._dispatch_expr(node.expr, frame, name).native, ddof=node.ddof
        )
        return self._with_native(result, name)

    def quantile(self, node: Quantile, frame: Frame, name: str) -> Scalar:
        result = fn.quantile(
            self._dispatch_expr(node.expr, frame, name).native,
            q=node.quantile,
            interpolation=node.interpolation,
        )[0]
        return self._with_native(result, name)

    def count(self, node: Count, frame: Frame, name: str) -> Scalar:
        result = fn.count(self._dispatch_expr(node.expr, frame, name).native)
        return self._with_native(result, name)

    def len(self, node: Len, frame: Frame, name: str) -> Scalar:
        result = fn.count(self._dispatch_expr(node.expr, frame, name).native, mode="all")
        return self._with_native(result, name)

    def max(self, node: Max, frame: Frame, name: str) -> Scalar:
        result: NativeScalar = fn.max_(self._dispatch_expr(node.expr, frame, name).native)
        return self._with_native(result, name)

    def mean(self, node: Mean, frame: Frame, name: str) -> Scalar:
        result = fn.mean(self._dispatch_expr(node.expr, frame, name).native)
        return self._with_native(result, name)

    def median(self, node: Median, frame: Frame, name: str) -> Scalar:
        result = fn.median(self._dispatch_expr(node.expr, frame, name).native)
        return self._with_native(result, name)

    def min(self, node: Min, frame: Frame, name: str) -> Scalar:
        result: NativeScalar = fn.min_(self._dispatch_expr(node.expr, frame, name).native)
        return self._with_native(result, name)

    # TODO @dangotbanned: top-level, complex-ish nodes
    # - [ ] Over
    #   - [x] `over_ordered`
    #   - [x] `group_by`, `join`
    #   - [x] `over` (with partitions)
    #   - [x] `over_ordered` (with partitions)
    #   - [ ] fix: join on nulls after https://github.com/narwhals-dev/narwhals/issues/3300
    # - [ ] `map_batches`
    #   - [x] elementwise
    #   - [ ] scalar
    # - [ ] `rolling_expr` has 4 variants

    def over(
        self,
        node: ir.WindowExpr,
        frame: Frame,
        name: str,
        *,
        reordered: Frame | None = None,
    ) -> Self:
        if is_function_expr(node.expr) and isinstance(
            node.expr.function, (IsFirstDistinct, IsLastDistinct)
        ):
            return self._is_first_last_distinct_partition_by(
                node.expr, frame, name, node.partition_by
            )
        # TODO @dangotbanned: Can the alias in `agg_irs` be avoided?
        resolved = (
            frame._grouper.by_irs(*node.partition_by)
            .agg_irs(node.expr.alias(name))
            .resolve(frame)
        )
        by_names = resolved.key_names
        windowed = resolved.evaluate(frame if reordered is None else reordered)
        return self.from_series(
            frame.select_names(*by_names)
            .join(windowed, how="left", left_on=by_names)
            .get_column(name)
        )

    def over_ordered(
        self, node: ir.OrderedWindowExpr, frame: Frame, name: str
    ) -> Self | Scalar:
        order_by = tuple(node.order_by_names())
        descending = node.sort_options.descending
        nulls_last = node.sort_options.nulls_last
        if node.partition_by:
            idx_name = temp.column_name(frame)
            if is_function_expr(node.expr) and isinstance(
                node.expr.function, (IsFirstDistinct, IsLastDistinct)
            ):
                frame = frame.with_row_index_by(
                    idx_name, order_by, descending=descending, nulls_last=nulls_last
                )
                expr_ir = fn.IS_FIRST_LAST_DISTINCT[type(node.expr.function)](idx_name)
                previous = node.expr.input[0].dispatch(self, frame, name)
                df = frame._with_columns([previous])
                distinct_index = (
                    df._grouper.by_irs(ir.col(name), *node.partition_by)
                    .agg_irs(expr_ir.alias(idx_name))
                    .resolve(df)
                    .evaluate(df)
                    .get_column(idx_name)
                    .native
                )
                return self._with_native(
                    fn.is_in(df.to_series().native, distinct_index), name
                )
            frame_indexed = frame.with_row_index_by(
                idx_name, order_by, nulls_last=nulls_last
            )
            reordered = frame_indexed.sort((idx_name,), descending=descending)
            return self.over(node, frame, name, reordered=reordered)
        opts = pa_options.sort(*order_by, descending=descending, nulls_last=nulls_last)
        indices = pc.sort_indices(frame.native, options=opts)
        evaluated = node.expr.dispatch(self, frame.gather(indices), name)
        if isinstance(evaluated, ArrowScalar):
            return evaluated
        return self.from_series(evaluated.broadcast(len(frame)).gather(indices))

    # NOTE: Can't implement in `EagerExpr`, since it doesn't derive `ExprDispatch`
    def map_batches(self, node: ir.AnonymousExpr, frame: Frame, name: str) -> Self:
        if node.is_scalar:
            # NOTE: Just trying to avoid redoing the whole API for `Series`
            msg = "Only elementwise is currently supported"
            raise NotImplementedError(msg)
        series = self._dispatch_expr(node.input[0], frame, name)
        udf = node.function.function
        result: Series | Into1DArray = udf(series)
        if not isinstance(result, Series):
            result = Series.from_numpy(result, name, version=self.version)
        if dtype := node.function.return_dtype:
            result = result.cast(dtype)
        return self.from_series(result)

    def rolling_expr(self, node: ir.RollingExpr, frame: Frame, name: str) -> Self:
        raise NotImplementedError

    def shift(self, node: FExpr[Shift], frame: Frame, name: str) -> Self:
        return self._function_expr(fn.shift, node.function.n)(node, frame, name)

    def diff(self, node: FExpr[Diff], frame: Frame, name: str) -> Self:
        return self._function_expr(fn.diff)(node, frame, name)

    def _cumulative(self, node: FExpr[CumAgg], frame: Frame, name: str) -> Self:
        return self._function_expr(fn.cumulative, node.function)(node, frame, name)

    cum_count = _cumulative
    cum_min = _cumulative
    cum_max = _cumulative
    cum_prod = _cumulative
    cum_sum = _cumulative

    def _is_first_last_distinct_partition_by(
        self,
        node: FExpr[IsFirstDistinct | IsLastDistinct],
        frame: Frame,
        name: str,
        partition_by: Seq[ir.ExprIR],
    ) -> Self:
        idx_name = temp.column_name([name])
        expr_ir = fn.IS_FIRST_LAST_DISTINCT[type(node.function)](idx_name)
        previous = node.input[0].dispatch(self, frame, name)
        df = frame._with_columns([previous]).with_row_index(idx_name)
        distinct_index = (
            df._grouper.by_irs(ir.col(name), *partition_by)
            .agg_irs(expr_ir.alias(idx_name))
            .resolve(df)
            .evaluate(df)
            .get_column(idx_name)
            .native
        )
        return self._with_native(fn.is_in(df.to_series().native, distinct_index), name)

    def _is_first_last_distinct(
        self, node: FExpr[IsFirstDistinct | IsLastDistinct], frame: Frame, name: str
    ) -> Self:
        idx_name = temp.column_name([name])
        expr_ir = fn.IS_FIRST_LAST_DISTINCT[type(node.function)](idx_name)
        series = self._dispatch_expr(node.input[0], frame, name)
        df = series.to_frame().with_row_index(idx_name)
        distinct_index = (
            df.group_by_names((name,))
            .agg((ir.named_ir(idx_name, expr_ir),))
            .get_column(idx_name)
            .native
        )
        return self._with_native(fn.is_in(df.to_series().native, distinct_index), name)

    is_first_distinct = _is_first_last_distinct
    is_last_distinct = _is_first_last_distinct

    def null_count(self, node: FExpr[NullCount], frame: Frame, name: str) -> Scalar:
        return self._function_expr(fn.null_count)(node, frame, name)

    def rank(self, node: FExpr[Rank], frame: Frame, name: str) -> Self:
        return self._function_expr(fn.rank, node.function.options)(node, frame, name)

    # ewm_mean = not_implemented()  # noqa: ERA001
    hist_bins = not_implemented()
    hist_bin_count = not_implemented()
    mode = not_implemented()
    unique = not_implemented()
    fill_null_with_strategy = not_implemented()
    kurtosis = not_implemented()
    skew = not_implemented()
    gather_every = not_implemented()
    is_duplicated = not_implemented()
    is_unique = not_implemented()


class ArrowScalar(
    _ArrowDispatch["ArrowScalar"],
    _StoresNative[NativeScalar],
    EagerScalar["Frame", Series],
):
    _evaluated: NativeScalar
    _version: Version
    _name: str

    @classmethod
    def from_native(
        cls,
        scalar: NativeScalar,
        name: str = "literal",
        /,
        version: Version = Version.MAIN,
    ) -> Self:
        obj = cls.__new__(cls)
        obj._evaluated = scalar
        obj._name = name
        obj._version = version
        return obj

    @classmethod
    def from_python(
        cls,
        value: PythonLiteral,
        name: str = "literal",
        /,
        *,
        dtype: IntoDType | None = None,
        version: Version = Version.MAIN,
    ) -> Self:
        dtype_pa: pa.DataType | None = None
        if dtype and dtype != version.dtypes.Unknown:
            dtype_pa = narwhals_to_native_dtype(dtype, version)
        return cls.from_native(fn.lit(value, dtype_pa), name, version)

    @classmethod
    def from_series(cls, series: Series) -> Self:
        if len(series) == 1:
            return cls.from_native(series.native[0], series.name, series.version)
        if len(series) == 0:
            return cls.from_python(
                None, series.name, dtype=series.dtype, version=series.version
            )
        msg = f"Too long {len(series)!r}"
        raise InvalidOperationError(msg)

    def _dispatch_expr(self, node: ir.ExprIR, frame: Frame, name: str) -> Series:
        msg = f"Expected unreachable, but hit at: {node!r}"
        raise InvalidOperationError(msg)

    def _with_native(self, native: Any, name: str, /) -> Self:
        return self.from_native(native, name or self.name, self.version)

    @property
    def native(self) -> NativeScalar:
        return self._evaluated

    def to_series(self) -> Series:
        return self.broadcast(1)

    def to_python(self) -> PythonLiteral:
        return self.native.as_py()  # type: ignore[no-any-return]

    def broadcast(self, length: int) -> Series:
        scalar = self.native
        if length == 1:
            chunked = fn.chunked_array(scalar)
        else:
            # NOTE: Same issue as `pa.scalar` overlapping overloads
            # https://github.com/zen-xu/pyarrow-stubs/pull/209
            pa_repeat: Incomplete = pa.repeat
            chunked = fn.chunked_array(pa_repeat(scalar, length))
        return Series.from_native(chunked, self.name, version=self.version)

    def count(self, node: Count, frame: Frame, name: str) -> Scalar:
        native = node.expr.dispatch(self, frame, name).native
        return self._with_native(pa.scalar(1 if native.is_valid else 0), name)

    def null_count(self, node: FExpr[NullCount], frame: Frame, name: str) -> Self:
        native = node.input[0].dispatch(self, frame, name).native
        return self._with_native(pa.scalar(0 if native.is_valid else 1), name)

    filter = not_implemented()
    over = not_implemented()
    over_ordered = not_implemented()
    map_batches = not_implemented()
    rank = not_implemented()
    # length_preserving
    rolling_expr = not_implemented()
    diff = not_implemented()
    cum_sum = not_implemented()  # TODO @dangotbanned: is this just self?
    cum_count = not_implemented()
    cum_min = not_implemented()
    cum_max = not_implemented()
    cum_prod = not_implemented()
