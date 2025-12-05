from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Protocol, TypeVar, overload

import pyarrow as pa  # ignore-banned-import
import pyarrow.compute as pc  # ignore-banned-import

from narwhals._arrow.utils import narwhals_to_native_dtype
from narwhals._plan import common, expressions as ir
from narwhals._plan._guards import (
    is_function_expr,
    is_iterable_reject,
    is_python_literal,
    is_seq_column,
)
from narwhals._plan.arrow import functions as fn
from narwhals._plan.arrow.series import ArrowSeries as Series
from narwhals._plan.arrow.typing import ChunkedOrScalarAny, NativeScalar, StoresNativeT_co
from narwhals._plan.common import temp
from narwhals._plan.compliant.accessors import (
    ExprCatNamespace,
    ExprListNamespace,
    ExprStringNamespace,
    ExprStructNamespace,
)
from narwhals._plan.compliant.column import ExprDispatch
from narwhals._plan.compliant.expr import EagerExpr
from narwhals._plan.compliant.scalar import EagerScalar
from narwhals._plan.compliant.typing import namespace
from narwhals._plan.expressions import FunctionExpr as FExpr, functions as F
from narwhals._plan.expressions.boolean import (
    IsDuplicated,
    IsFirstDistinct,
    IsInExpr,
    IsInSeq,
    IsInSeries,
    IsLastDistinct,
    IsNotNan,
    IsNotNull,
    IsUnique,
)
from narwhals._plan.expressions.functions import NullCount
from narwhals._utils import (
    Implementation,
    Version,
    _StoresNative,
    not_implemented,
    qualified_type_name,
)
from narwhals.exceptions import InvalidOperationError, ShapeError

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from typing_extensions import Self, TypeAlias

    from narwhals._plan.arrow.dataframe import ArrowDataFrame as Frame
    from narwhals._plan.arrow.namespace import ArrowNamespace
    from narwhals._plan.arrow.typing import (
        ChunkedArrayAny,
        P,
        UnaryFunctionP,
        VectorFunction,
    )
    from narwhals._plan.expressions import (
        BinaryExpr,
        FunctionExpr as FExpr,
        lists,
        strings,
    )
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
    from narwhals._plan.expressions.categorical import GetCategories
    from narwhals._plan.expressions.functions import (
        Abs,
        CumAgg,
        Diff,
        FillNan,
        FillNull,
        NullCount,
        Pow,
        Rank,
        Shift,
    )
    from narwhals._plan.expressions.struct import FieldByName
    from narwhals._plan.typing import Seq
    from narwhals.typing import IntoDType, PythonLiteral

    Expr: TypeAlias = "ArrowExpr"
    Scalar: TypeAlias = "ArrowScalar"


BACKEND_VERSION = Implementation.PYARROW._backend_version()


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
        return self._with_native(fn.power(base_, exponent_), name)

    def fill_null(
        self, node: FExpr[FillNull], frame: Frame, name: str
    ) -> StoresNativeT_co:
        expr, value = node.function.unwrap_input(node)
        native = expr.dispatch(self, frame, name).native
        value_ = value.dispatch(self, frame, "value").native
        return self._with_native(pc.fill_null(native, value_), name)

    def fill_nan(self, node: FExpr[FillNan], frame: Frame, name: str) -> StoresNativeT_co:
        expr, value = node.function.unwrap_input(node)
        native = expr.dispatch(self, frame, name).native
        value_ = value.dispatch(self, frame, "value").native
        return self._with_native(fn.fill_nan(native, value_), name)

    def is_between(
        self, node: FExpr[IsBetween], frame: Frame, name: str
    ) -> StoresNativeT_co:
        expr, lower_bound, upper_bound = node.function.unwrap_input(node)
        native = expr.dispatch(self, frame, name).native
        lower = lower_bound.dispatch(self, frame, "lower").native
        upper = upper_bound.dispatch(self, frame, "upper").native
        result = fn.is_between(native, lower, upper, node.function.closed)
        return self._with_native(result, name)

    @overload
    def _unary_function(
        self, fn_native: UnaryFunctionP[P], /, *args: P.args, **kwds: P.kwargs
    ) -> Callable[[FExpr[Any], Frame, str], StoresNativeT_co]: ...
    @overload
    def _unary_function(
        self, fn_native: Callable[[ChunkedOrScalarAny], ChunkedOrScalarAny], /
    ) -> Callable[[FExpr[Any], Frame, str], StoresNativeT_co]: ...
    def _unary_function(
        self, fn_native: UnaryFunctionP[P], /, *args: P.args, **kwds: P.kwargs
    ) -> Callable[[FExpr[Any], Frame, str], StoresNativeT_co]:
        """Return a function with the signature `(node, frame, name)`.

        Handles dispatching prior expressions, and rewrapping the result of this one.

        Arity refers to the number of expression inputs to a function (after expanding).

        So a **unary** function will look like:

            col("a").round(2)

        Which unravels to:

            FunctionExpr(
                input=(Column(name="a"),),
                #                      ^ length-1 tuple
                function=Round(decimals=2),
                #                       ^ non-expression argument
                options=...,
            )
        """

        def func(node: FExpr[Any], frame: Frame, name: str, /) -> StoresNativeT_co:
            native = node.input[0].dispatch(self, frame, name).native
            return self._with_native(fn_native(native, *args, **kwds), name)

        return func

    def abs(self, node: FExpr[Abs], frame: Frame, name: str) -> StoresNativeT_co:
        return self._unary_function(fn.abs_)(node, frame, name)

    def not_(self, node: FExpr[Not], frame: Frame, name: str) -> StoresNativeT_co:
        return self._unary_function(fn.not_)(node, frame, name)

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

    def is_in_expr(
        self, node: FExpr[IsInExpr], frame: Frame, name: str
    ) -> StoresNativeT_co:
        expr, other = node.function.unwrap_input(node)
        right = other.dispatch(self, frame, name).native
        arr = fn.array(right) if isinstance(right, pa.Scalar) else right
        result = fn.is_in(expr.dispatch(self, frame, name).native, arr)
        return self._with_native(result, name)

    def is_in_series(
        self, node: FExpr[IsInSeries[ChunkedArrayAny]], frame: Frame, name: str
    ) -> StoresNativeT_co:
        other = node.function.other.unwrap().to_native()
        return self._unary_function(fn.is_in, other)(node, frame, name)

    def is_in_seq(
        self, node: FExpr[IsInSeq], frame: Frame, name: str
    ) -> StoresNativeT_co:
        other = fn.array(node.function.other)
        return self._unary_function(fn.is_in, other)(node, frame, name)

    def is_nan(self, node: FExpr[IsNan], frame: Frame, name: str) -> StoresNativeT_co:
        return self._unary_function(fn.is_nan)(node, frame, name)

    def is_null(self, node: FExpr[IsNull], frame: Frame, name: str) -> StoresNativeT_co:
        return self._unary_function(fn.is_null)(node, frame, name)

    def is_not_nan(
        self, node: FExpr[IsNotNan], frame: Frame, name: str
    ) -> StoresNativeT_co:
        return self._unary_function(fn.is_not_nan)(node, frame, name)

    def is_not_null(
        self, node: FExpr[IsNotNull], frame: Frame, name: str
    ) -> StoresNativeT_co:
        return self._unary_function(fn.is_not_null)(node, frame, name)

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

    def log(self, node: FExpr[F.Log], frame: Frame, name: str) -> StoresNativeT_co:
        return self._unary_function(fn.log, node.function.base)(node, frame, name)

    def exp(self, node: FExpr[F.Exp], frame: Frame, name: str) -> StoresNativeT_co:
        return self._unary_function(fn.exp)(node, frame, name)

    def sqrt(self, node: FExpr[F.Sqrt], frame: Frame, name: str) -> StoresNativeT_co:
        return self._unary_function(fn.sqrt)(node, frame, name)

    def round(self, node: FExpr[F.Round], frame: Frame, name: str) -> StoresNativeT_co:
        return self._unary_function(fn.round, node.function.decimals)(node, frame, name)

    def ceil(self, node: FExpr[F.Ceil], frame: Frame, name: str) -> StoresNativeT_co:
        return self._unary_function(fn.ceil)(node, frame, name)

    def floor(self, node: FExpr[F.Floor], frame: Frame, name: str) -> StoresNativeT_co:
        return self._unary_function(fn.floor)(node, frame, name)

    def clip(self, node: FExpr[F.Clip], frame: Frame, name: str) -> StoresNativeT_co:
        expr, lower, upper = node.function.unwrap_input(node)
        result = fn.clip(
            expr.dispatch(self, frame, name).native,
            lower.dispatch(self, frame, name).native,
            upper.dispatch(self, frame, name).native,
        )
        return self._with_native(result, name)

    def clip_lower(
        self, node: FExpr[F.ClipLower], frame: Frame, name: str
    ) -> StoresNativeT_co:
        expr, other = node.function.unwrap_input(node)
        result = fn.clip_lower(
            expr.dispatch(self, frame, name).native,
            other.dispatch(self, frame, name).native,
        )
        return self._with_native(result, name)

    def clip_upper(
        self, node: FExpr[F.ClipUpper], frame: Frame, name: str
    ) -> StoresNativeT_co:
        expr, other = node.function.unwrap_input(node)
        result = fn.clip_upper(
            expr.dispatch(self, frame, name).native,
            other.dispatch(self, frame, name).native,
        )
        return self._with_native(result, name)

    def replace_strict(
        self, node: FExpr[F.ReplaceStrict], frame: Frame, name: str
    ) -> StoresNativeT_co:
        old, new = node.function.old, node.function.new
        dtype = fn.dtype_native(node.function.return_dtype, self.version)
        return self._unary_function(fn.replace_strict, old, new, dtype)(node, frame, name)

    def replace_strict_default(
        self, node: FExpr[F.ReplaceStrictDefault], frame: Frame, name: str
    ) -> StoresNativeT_co:
        func = node.function
        expr, default_ = func.unwrap_input(node)
        native = expr.dispatch(self, frame, name).native
        default = default_.dispatch(self, frame, name).native
        dtype = fn.dtype_native(func.return_dtype, self.version)
        result = fn.replace_strict_default(native, func.old, func.new, default, dtype)
        return self._with_native(result, name)


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

    def _vector_function(
        self, fn_native: VectorFunction[P], *args: P.args, **kwds: P.kwargs
    ) -> Callable[[FExpr[Any], Frame, str], Self]:
        def func(node: FExpr[Any], frame: Frame, name: str, /) -> Self:  # type: ignore[type-var, misc]
            native = self._dispatch_expr(node.input[0], frame, name).native
            return self._with_native(fn_native(native, *args, **kwds), name)

        return func

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
        indices = fn.sort_indices(df.native, *keys, options=node.options)
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

    def null_count(self, node: FExpr[F.NullCount], frame: Frame, name: str) -> Scalar:
        native = self._dispatch_expr(node.input[0], frame, name).native
        return self._with_native(fn.null_count(native), name)

    def kurtosis(self, node: FExpr[F.Kurtosis], frame: Frame, name: str) -> Scalar:
        native = self._dispatch_expr(node.input[0], frame, name).native
        return self._with_native(fn.kurtosis_skew(native, "kurtosis"), name)

    def skew(self, node: FExpr[F.Skew], frame: Frame, name: str) -> Scalar:
        native = self._dispatch_expr(node.input[0], frame, name).native
        return self._with_native(fn.kurtosis_skew(native, "skew"), name)

    def over(
        self,
        node: ir.WindowExpr,
        frame: Frame,
        name: str,
        *,
        sort_indices: pa.UInt64Array | None = None,
    ) -> Self:
        expr = node.expr
        by = node.partition_by
        if is_function_expr(expr) and isinstance(
            expr.function, (IsFirstDistinct, IsLastDistinct, IsUnique, IsDuplicated)
        ):
            return self._boolean_length_preserving(
                expr, frame, name, by, sort_indices=sort_indices
            )
        resolved = frame._grouper.by_irs(*by).agg_irs(expr.alias(name)).resolve(frame)
        results = frame.group_by_resolver(resolved).agg_over(resolved.aggs, sort_indices)
        return self.from_series(results.get_column(name))

    def over_ordered(
        self, node: ir.OrderedWindowExpr, frame: Frame, name: str
    ) -> Self | Scalar:
        by = node.order_by_names()
        indices = fn.sort_indices(frame.native, *by, options=node.sort_options)
        if node.partition_by:
            return self.over(node, frame, name, sort_indices=indices)
        evaluated = node.expr.dispatch(self, frame.gather(indices), name)
        if isinstance(evaluated, ArrowScalar):
            return evaluated
        return self.from_series(evaluated.broadcast(len(frame)).gather(indices))

    def _boolean_length_preserving(
        self,
        node: FExpr[IsFirstDistinct | IsLastDistinct | IsUnique | IsDuplicated],
        frame: Frame,
        name: str,
        partition_by: Seq[ir.ExprIR] = (),
        *,
        sort_indices: pa.UInt64Array | None = None,
    ) -> Self:
        # NOTE: This subset of functions can be expressed as a mask applied to indices
        into_column_agg, mask = fn.BOOLEAN_LENGTH_PRESERVING[type(node.function)]
        idx_name = temp.column_name(frame)
        df = frame._with_columns([node.input[0].dispatch(self, frame, name)])
        if sort_indices is not None:
            column = fn.unsort_indices(sort_indices)
            df = df._with_native(df.native.add_column(0, idx_name, column))
        else:
            df = df.with_row_index(idx_name)
        agg_node = into_column_agg(idx_name)
        if not (partition_by or sort_indices is not None):
            aggregated = df.group_by_names((name,)).agg(
                (ir.named_ir(idx_name, agg_node),)
            )
        else:
            aggregated = df.group_by_agg_irs((ir.col(name), *partition_by), agg_node)
        index = df.to_series().alias(name)
        final_result = mask(index.native, aggregated.get_column(idx_name).native)
        return self.from_series(index._with_native(final_result))

    # NOTE: Can't implement in `EagerExpr` (like on `main`)
    # The version here is missing `__narwhals_namespace__`
    def map_batches(
        self, node: ir.AnonymousExpr, frame: Frame, name: str
    ) -> Self | Scalar:
        series = self._dispatch_expr(node.input[0], frame, name)
        udf = node.function.function
        udf_result: Series | Iterable[Any] | Any = udf(series)
        if node.is_scalar:
            return ArrowScalar.from_unknown(
                udf_result, name, dtype=node.function.return_dtype, version=self.version
            )
        if isinstance(udf_result, Series):
            result = udf_result
        elif isinstance(udf_result, Iterable) and not is_iterable_reject(udf_result):
            result = Series.from_iterable(udf_result, name=name, version=self.version)
        else:
            msg = (
                "`map_batches` with `returns_scalar=False` must return a Series; "
                f"found '{qualified_type_name(udf_result)}'.\n\nIf `returns_scalar` "
                "is set to `True`, a returned value can be a scalar value."
            )
            raise TypeError(msg)
        if dtype := node.function.return_dtype:
            result = result.cast(dtype)
        return self.from_series(result)

    def shift(self, node: FExpr[Shift], frame: Frame, name: str) -> Self:
        return self._vector_function(fn.shift, node.function.n)(node, frame, name)

    def diff(self, node: FExpr[Diff], frame: Frame, name: str) -> Self:
        return self._vector_function(fn.diff)(node, frame, name)

    def rank(self, node: FExpr[Rank], frame: Frame, name: str) -> Self:
        return self._vector_function(fn.rank, node.function.options)(node, frame, name)

    def _cumulative(self, node: FExpr[CumAgg], frame: Frame, name: str) -> Self:
        native = self._dispatch_expr(node.input[0], frame, name).native
        return self._with_native(fn.cumulative(native, node.function), name)

    def unique(self, node: FExpr[F.Unique], frame: Frame, name: str) -> Self:
        return self.from_series(self._dispatch_expr(node.input[0], frame, name).unique())

    def gather_every(self, node: FExpr[F.GatherEvery], frame: Frame, name: str) -> Self:
        series = self._dispatch_expr(node.input[0], frame, name)
        n, offset = node.function.n, node.function.offset
        return self.from_series(series.gather_every(n=n, offset=offset))

    def sample_n(self, node: FExpr[F.SampleN], frame: Frame, name: str) -> Self:
        series = self._dispatch_expr(node.input[0], frame, name)
        func = node.function
        n, replace, seed = func.n, func.with_replacement, func.seed
        result = series.sample_n(n, with_replacement=replace, seed=seed)
        return self.from_series(result)

    def sample_frac(self, node: FExpr[F.SampleFrac], frame: Frame, name: str) -> Self:
        series = self._dispatch_expr(node.input[0], frame, name)
        func = node.function
        fraction, replace, seed = func.fraction, func.with_replacement, func.seed
        result = series.sample_frac(fraction, with_replacement=replace, seed=seed)
        return self.from_series(result)

    def drop_nulls(self, node: FExpr[F.DropNulls], frame: Frame, name: str) -> Self:
        return self._vector_function(fn.drop_nulls)(node, frame, name)

    def mode_all(self, node: FExpr[F.ModeAll], frame: Frame, name: str) -> Self:
        return self._vector_function(fn.mode_all)(node, frame, name)

    def mode_any(self, node: FExpr[F.ModeAny], frame: Frame, name: str) -> Scalar:
        native = self._dispatch_expr(node.input[0], frame, name).native
        return self._with_native(fn.mode_any(native), name)

    def fill_null_with_strategy(
        self, node: FExpr[F.FillNullWithStrategy], frame: Frame, name: str
    ) -> Self:
        native = self._dispatch_expr(node.input[0], frame, name).native
        strategy, limit = node.function.strategy, node.function.limit
        func = fn.fill_null_with_strategy
        return self._with_native(func(native, strategy, limit), name)

    cum_count = _cumulative
    cum_min = _cumulative
    cum_max = _cumulative
    cum_prod = _cumulative
    cum_sum = _cumulative
    is_first_distinct = _boolean_length_preserving
    is_last_distinct = _boolean_length_preserving
    is_duplicated = _boolean_length_preserving
    is_unique = _boolean_length_preserving

    _ROLLING: ClassVar[Mapping[type[F.RollingWindow], Callable[..., Series]]] = {
        F.RollingSum: Series.rolling_sum,
        F.RollingMean: Series.rolling_mean,
        F.RollingVar: Series.rolling_var,
        F.RollingStd: Series.rolling_std,
    }

    def rolling_expr(
        self, node: ir.RollingExpr[F.RollingWindow], frame: Frame, name: str
    ) -> Self:
        s = self._dispatch_expr(node.input[0], frame, name)
        roll_options = node.function.options
        size = roll_options.window_size
        samples = roll_options.min_samples
        center = roll_options.center
        op = type(node.function)
        method = self._ROLLING[op]
        if op in {F.RollingSum, F.RollingMean}:
            return self.from_series(method(s, size, min_samples=samples, center=center))
        ddof = roll_options.ddof
        result = method(s, size, min_samples=samples, center=center, ddof=ddof)
        return self.from_series(result)

    def hist_bins(self, node: FExpr[F.HistBins], frame: Frame, name: str) -> Self:
        native = self._dispatch_expr(node.input[0], frame, name).native
        bins = list(node.function.bins)
        include = node.function.include_breakpoint
        if len(bins) <= 1:
            data = fn._hist_data_empty(include_breakpoint=include)
        elif fn.is_only_nulls(native, nan_is_null=True):
            data = fn._hist_series_empty(bins, include_breakpoint=include)
        else:
            data = fn._hist_calculate_hist(native, bins, include_breakpoint=include)
        ns = namespace(self)
        return self.from_series(ns._dataframe.from_dict(data).to_struct(name))

    def hist_bin_count(
        self, node: FExpr[F.HistBinCount], frame: Frame, name: str
    ) -> Self:
        s = self._dispatch_expr(node.input[0], frame, name)
        func = node.function
        struct_data = fn.hist_with_bin_count(
            s.native, func.bin_count, include_breakpoint=func.include_breakpoint
        )
        return self.from_series(
            namespace(self)._dataframe.from_dict(struct_data).to_struct(name)
        )

    # ewm_mean = not_implemented()  # noqa: ERA001
    @property
    def cat(self) -> ArrowCatNamespace[Expr]:
        return ArrowCatNamespace(self)

    @property
    def list(self) -> ArrowListNamespace[Expr]:
        return ArrowListNamespace(self)

    @property
    def str(self) -> ArrowStringNamespace[Expr]:
        return ArrowStringNamespace(self)

    @property
    def struct(self) -> ArrowStructNamespace[Expr]:
        return ArrowStructNamespace(self)


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

    @classmethod
    def from_unknown(
        cls,
        value: Any,
        name: str = "literal",
        /,
        *,
        dtype: IntoDType | None = None,
        version: Version = Version.MAIN,
    ) -> Self:
        if isinstance(value, pa.Scalar):
            return cls.from_native(value, name, version)
        if is_python_literal(value):
            return cls.from_python(value, name, dtype=dtype, version=version)
        native = fn.lit(value, fn.dtype_native(dtype, version))
        return cls.from_native(native, name, version)

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
        result: PythonLiteral = self.native.as_py()
        return result

    def broadcast(self, length: int) -> Series:
        scalar = self.native
        if length == 1:
            chunked = fn.chunked_array(scalar)
        else:
            chunked = fn.chunked_array(fn.repeat_unchecked(scalar, length))
        return Series.from_native(chunked, self.name, version=self.version)

    def count(self, node: Count, frame: Frame, name: str) -> Scalar:
        native = node.expr.dispatch(self, frame, name).native
        return self._with_native(pa.scalar(1 if native.is_valid else 0), name)

    def null_count(self, node: FExpr[NullCount], frame: Frame, name: str) -> Self:
        native = node.input[0].dispatch(self, frame, name).native
        return self._with_native(pa.scalar(0 if native.is_valid else 1), name)

    def drop_nulls(  # type: ignore[override]
        self, node: FExpr[F.DropNulls], frame: Frame, name: str
    ) -> Scalar | Expr:
        previous = node.input[0].dispatch(self, frame, name)
        if previous.native.is_valid:
            return previous
        chunked = fn.chunked_array([[]], previous.native.type)
        return ArrowExpr.from_native(chunked, name, version=self.version)

    @property
    def cat(self) -> ArrowCatNamespace[Scalar]:
        return ArrowCatNamespace(self)

    @property
    def list(self) -> ArrowListNamespace[Scalar]:
        return ArrowListNamespace(self)

    @property
    def str(self) -> ArrowStringNamespace[Scalar]:
        return ArrowStringNamespace(self)

    @property
    def struct(self) -> ArrowStructNamespace[Scalar]:
        return ArrowStructNamespace(self)

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


ExprOrScalarT = TypeVar("ExprOrScalarT", ArrowExpr, ArrowScalar)


class ArrowAccessor(Generic[ExprOrScalarT]):
    def __init__(self, compliant: ExprOrScalarT, /) -> None:
        self._compliant: ExprOrScalarT = compliant

    @property
    def compliant(self) -> ExprOrScalarT:
        return self._compliant

    def __narwhals_namespace__(self) -> ArrowNamespace:
        return namespace(self.compliant)

    @property
    def version(self) -> Version:
        return self.compliant.version

    def with_native(self, native: ChunkedOrScalarAny, name: str, /) -> Expr | Scalar:
        return self.compliant._with_native(native, name)

    @overload
    def unary(
        self, fn_native: UnaryFunctionP[P], /, *args: P.args, **kwds: P.kwargs
    ) -> Callable[[FExpr[Any], Frame, str], Expr | Scalar]: ...
    @overload
    def unary(
        self, fn_native: Callable[[ChunkedOrScalarAny], ChunkedOrScalarAny], /
    ) -> Callable[[FExpr[Any], Frame, str], Expr | Scalar]: ...
    def unary(
        self, fn_native: UnaryFunctionP[P], /, *args: P.args, **kwds: P.kwargs
    ) -> Callable[[FExpr[Any], Frame, str], Expr | Scalar]:
        return self.compliant._unary_function(fn_native, *args, **kwds)


class ArrowCatNamespace(ExprCatNamespace["Frame", "Expr"], ArrowAccessor[ExprOrScalarT]):
    def get_categories(self, node: FExpr[GetCategories], frame: Frame, name: str) -> Expr:
        native = node.input[0].dispatch(self.compliant, frame, name).native
        return ArrowExpr.from_native(fn.get_categories(native), name, self.version)


class ArrowListNamespace(
    ExprListNamespace["Frame", "Expr | Scalar"], ArrowAccessor[ExprOrScalarT]
):
    def len(self, node: FExpr[lists.Len], frame: Frame, name: str) -> Expr | Scalar:
        return self.unary(fn.list_len)(node, frame, name)

    def get(self, node: FExpr[lists.Get], frame: Frame, name: str) -> Expr | Scalar:
        return self.unary(fn.list_get, node.function.index)(node, frame, name)

    unique = not_implemented()
    contains = not_implemented()


# TODO @dangotbanned: Add tests for these, especially those using a different native function
class ArrowStringNamespace(
    ExprStringNamespace["Frame", "Expr | Scalar"], ArrowAccessor[ExprOrScalarT]
):
    def len_chars(
        self, node: FExpr[strings.LenChars], frame: Frame, name: str
    ) -> Expr | Scalar:
        return self.unary(fn.str_len_chars)(node, frame, name)

    def slice(self, node: FExpr[strings.Slice], frame: Frame, name: str) -> Expr | Scalar:
        offset, length = node.function.offset, node.function.length
        return self.unary(fn.str_slice, offset, length)(node, frame, name)

    def zfill(self, node: FExpr[strings.ZFill], frame: Frame, name: str) -> Expr | Scalar:
        return self.unary(fn.str_zfill, node.function.length)(node, frame, name)

    def contains(
        self, node: FExpr[strings.Contains], frame: Frame, name: str
    ) -> Expr | Scalar:
        pattern, literal = node.function.pattern, node.function.literal
        return self.unary(fn.str_contains, pattern, literal=literal)(node, frame, name)

    def ends_with(
        self, node: FExpr[strings.EndsWith], frame: Frame, name: str
    ) -> Expr | Scalar:
        return self.unary(fn.str_ends_with, node.function.suffix)(node, frame, name)

    def replace(
        self, node: FExpr[strings.Replace], frame: Frame, name: str
    ) -> Expr | Scalar:
        func = node.function
        pattern, literal, n = (func.pattern, func.literal, func.n)
        expr, other = func.unwrap_input(node)
        prev = expr.dispatch(self.compliant, frame, name)
        value = other.dispatch(self.compliant, frame, name)
        if isinstance(value, ArrowScalar):
            result = fn.str_replace(
                prev.native, pattern, value.native.as_py(), literal=literal, n=n
            )
        elif isinstance(prev, ArrowExpr):
            result = fn.str_replace_vector(
                prev.native, pattern, value.native, literal=literal, n=n
            )
        else:
            # not sure this even makes sense
            msg = "TODO: `ArrowScalar.str.replace(value: ArrowExpr)`"
            raise NotImplementedError(msg)
        return self.with_native(result, name)

    def replace_all(
        self, node: FExpr[strings.ReplaceAll], frame: Frame, name: str
    ) -> Expr | Scalar:
        rewrite: FExpr[Any] = common.replace(
            node, function=node.function.to_replace_n(-1)
        )
        return self.replace(rewrite, frame, name)

    def split(self, node: FExpr[strings.Split], frame: Frame, name: str) -> Expr | Scalar:
        return self.unary(fn.str_split, node.function.by)(node, frame, name)

    def starts_with(
        self, node: FExpr[strings.StartsWith], frame: Frame, name: str
    ) -> Expr | Scalar:
        return self.unary(fn.str_starts_with, node.function.prefix)(node, frame, name)

    def strip_chars(
        self, node: FExpr[strings.StripChars], frame: Frame, name: str
    ) -> Expr | Scalar:
        return self.unary(fn.str_strip_chars, node.function.characters)(node, frame, name)

    def to_uppercase(
        self, node: FExpr[strings.ToUppercase], frame: Frame, name: str
    ) -> Expr | Scalar:
        return self.unary(fn.str_to_uppercase)(node, frame, name)

    def to_lowercase(
        self, node: FExpr[strings.ToLowercase], frame: Frame, name: str
    ) -> Expr | Scalar:
        return self.unary(fn.str_to_lowercase)(node, frame, name)

    def to_titlecase(
        self, node: FExpr[strings.ToTitlecase], frame: Frame, name: str
    ) -> Expr | Scalar:
        return self.unary(fn.str_to_titlecase)(node, frame, name)

    to_date = not_implemented()
    to_datetime = not_implemented()


class ArrowStructNamespace(
    ExprStructNamespace["Frame", "Expr | Scalar"], ArrowAccessor[ExprOrScalarT]
):
    def field(self, node: FExpr[FieldByName], frame: Frame, name: str) -> Expr | Scalar:
        return self.unary(fn.struct_field, node.function.name)(node, frame, name)
