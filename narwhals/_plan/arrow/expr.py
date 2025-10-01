from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, overload

import pyarrow as pa  # ignore-banned-import
import pyarrow.compute as pc  # ignore-banned-import

from narwhals._arrow.utils import narwhals_to_native_dtype
from narwhals._plan.arrow import functions as fn
from narwhals._plan.arrow.series import ArrowSeries as Series
from narwhals._plan.arrow.typing import ChunkedOrScalarAny, NativeScalar, StoresNativeT_co
from narwhals._plan.compliant.typing import namespace
from narwhals._plan.expressions import NamedIR
from narwhals._plan.protocols import EagerExpr, EagerScalar, ExprDispatch
from narwhals._utils import (
    Implementation,
    Version,
    _StoresNative,
    generate_temporary_column_name,
    not_implemented,
)
from narwhals.exceptions import InvalidOperationError, ShapeError

if TYPE_CHECKING:
    from collections.abc import Callable

    from typing_extensions import Self, TypeAlias

    from narwhals._arrow.typing import ChunkedArrayAny, Incomplete
    from narwhals._plan import expressions as ir
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
    from narwhals._plan.expressions.expr import BinaryExpr, FunctionExpr
    from narwhals._plan.expressions.functions import Abs, FillNull, Pow
    from narwhals.typing import Into1DArray, IntoDType, PythonLiteral

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

    def pow(self, node: FunctionExpr[Pow], frame: Frame, name: str) -> StoresNativeT_co:
        base, exponent = node.function.unwrap_input(node)
        base_ = base.dispatch(self, frame, "base").native
        exponent_ = exponent.dispatch(self, frame, "exponent").native
        return self._with_native(pc.power(base_, exponent_), name)

    def fill_null(
        self, node: FunctionExpr[FillNull], frame: Frame, name: str
    ) -> StoresNativeT_co:
        expr, value = node.function.unwrap_input(node)
        native = expr.dispatch(self, frame, name).native
        value_ = value.dispatch(self, frame, "value").native
        return self._with_native(pc.fill_null(native, value_), name)

    def is_between(
        self, node: FunctionExpr[IsBetween], frame: Frame, name: str
    ) -> StoresNativeT_co:
        expr, lower_bound, upper_bound = node.function.unwrap_input(node)
        native = expr.dispatch(self, frame, name).native
        lower = lower_bound.dispatch(self, frame, "lower").native
        upper = upper_bound.dispatch(self, frame, "upper").native
        result = fn.is_between(native, lower, upper, node.function.closed)
        return self._with_native(result, name)

    def _unary_function(
        self, fn_native: Callable[[Any], Any], /
    ) -> Callable[[FunctionExpr[Any], Frame, str], StoresNativeT_co]:
        def func(node: FunctionExpr[Any], frame: Frame, name: str) -> StoresNativeT_co:
            native = node.input[0].dispatch(self, frame, name).native
            return self._with_native(fn_native(native), name)

        return func

    def abs(self, node: FunctionExpr[Abs], frame: Frame, name: str) -> StoresNativeT_co:
        return self._unary_function(pc.abs)(node, frame, name)

    def not_(self, node: FunctionExpr[Not], frame: Frame, name: str) -> StoresNativeT_co:
        return self._unary_function(pc.invert)(node, frame, name)

    def all(self, node: FunctionExpr[All], frame: Frame, name: str) -> StoresNativeT_co:
        return self._unary_function(fn.all_)(node, frame, name)

    def any(
        self, node: FunctionExpr[ir.boolean.Any], frame: Frame, name: str
    ) -> StoresNativeT_co:
        return self._unary_function(fn.any_)(node, frame, name)

    def is_finite(
        self, node: FunctionExpr[IsFinite], frame: Frame, name: str
    ) -> StoresNativeT_co:
        return self._unary_function(fn.is_finite)(node, frame, name)

    def is_nan(
        self, node: FunctionExpr[IsNan], frame: Frame, name: str
    ) -> StoresNativeT_co:
        return self._unary_function(fn.is_nan)(node, frame, name)

    def is_null(
        self, node: FunctionExpr[IsNull], frame: Frame, name: str
    ) -> StoresNativeT_co:
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

    def _dispatch_expr(self, node: ir.ExprIR, frame: Frame, name: str) -> Series:
        """Use instead of `_dispatch` *iff* an operation isn't natively supported on `ChunkedArray`.

        There is no need to broadcast, as they may have a cheaper impl elsewhere (`CompliantScalar` or `ArrowScalar`).

        Mainly for the benefit of a type checker, but the equivalent `ArrowScalar._dispatch_expr` will raise if
        the assumption fails.
        """
        return node.dispatch(self, frame, name).to_series()

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
        native = self._dispatch_expr(node.expr, frame, name).native
        sorted_indices = pc.array_sort_indices(native, options=node.options.to_arrow())
        return self._with_native(native.take(sorted_indices), name)

    def sort_by(self, node: ir.SortBy, frame: Frame, name: str) -> Expr:
        series = self._dispatch_expr(node.expr, frame, name)
        by = (
            self._dispatch_expr(e, frame, f"<TEMP>_{idx}")
            for idx, e in enumerate(node.by)
        )
        df = namespace(self)._concat_horizontal((series, *by))
        names = df.columns[1:]
        indices = pc.sort_indices(df.native, options=node.options.to_arrow(names))
        result: ChunkedArrayAny = df.native.column(0).take(indices)
        return self._with_native(result, name)

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
    # - [ ] `over`/`_ordered` (with partitions) requires `group_by`, `join`
    # - [x] `over_ordered` alone should be possible w/ the current API
    # - [x] `map_batches` is defined in `EagerExpr`, might be simpler here than on main
    # - [ ] `rolling_expr` has 4 variants

    def over(self, node: ir.WindowExpr, frame: Frame, name: str) -> Self:
        raise NotImplementedError

    def over_ordered(
        self, node: ir.OrderedWindowExpr, frame: Frame, name: str
    ) -> Self | Scalar:
        if node.partition_by:
            msg = f"Need to implement `group_by`, `join` for:\n{node!r}"
            raise NotImplementedError(msg)

        # NOTE: Converting `over(order_by=..., options=...)` into the right shape for `DataFrame.sort`
        sort_by = tuple(NamedIR.from_ir(e) for e in node.order_by)
        options = node.sort_options.to_multiple(len(node.order_by))
        idx_name = generate_temporary_column_name(8, frame.columns)
        sorted_context = frame.with_row_index(idx_name).sort(sort_by, options)
        evaluated = node.expr.dispatch(self, sorted_context.drop([idx_name]), name)
        if isinstance(evaluated, ArrowScalar):
            # NOTE: We're already sorted, defer broadcasting to the outer context
            # Wouldn't be suitable for partitions, but will be fine here
            # - https://github.com/narwhals-dev/narwhals/pull/2528/commits/2ae42458cae91f4473e01270919815fcd7cb9667
            # - https://github.com/narwhals-dev/narwhals/pull/2528/commits/b8066c4c57d4b0b6c38d58a0f5de05eefc2cae70
            return self._with_native(evaluated.native, name)
        indices = pc.sort_indices(sorted_context.get_column(idx_name).native)
        height = len(sorted_context)
        result = evaluated.broadcast(height).native.take(indices)
        return self._with_native(result, name)

    # NOTE: Can't implement in `EagerExpr`, since it doesn't derive `ExprDispatch`
    def map_batches(self, node: ir.AnonymousExpr, frame: Frame, name: str) -> Self:
        if node.is_scalar:
            # NOTE: Just trying to avoid redoing the whole API for `Series`
            msg = "Only elementwise is currently supported"
            raise NotImplementedError(msg)
        series = self._dispatch_expr(node.input[0], frame, name)
        udf = node.function.function
        result: Series | Into1DArray = udf(series)
        if not fn.is_series(result):
            result = Series.from_numpy(result, name, version=self.version)
        if dtype := node.function.return_dtype:
            result = result.cast(dtype)
        return self.from_series(result)

    def rolling_expr(self, node: ir.RollingExpr, frame: Frame, name: str) -> Self:
        raise NotImplementedError


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

    def arg_min(self, node: ArgMin, frame: Frame, name: str) -> Scalar:
        return self._with_native(pa.scalar(0), name)

    def arg_max(self, node: ArgMax, frame: Frame, name: str) -> Scalar:
        return self._with_native(pa.scalar(0), name)

    def n_unique(self, node: NUnique, frame: Frame, name: str) -> Scalar:
        return self._with_native(pa.scalar(1), name)

    def std(self, node: Std, frame: Frame, name: str) -> Scalar:
        return self._with_native(pa.scalar(None, pa.null()), name)

    def var(self, node: Var, frame: Frame, name: str) -> Scalar:
        return self._with_native(pa.scalar(None, pa.null()), name)

    def count(self, node: Count, frame: Frame, name: str) -> Scalar:
        native = node.expr.dispatch(self, frame, name).native
        return self._with_native(pa.scalar(1 if native.is_valid else 0), name)

    def len(self, node: Len, frame: Frame, name: str) -> Scalar:
        return self._with_native(pa.scalar(1), name)

    filter = not_implemented()
    over = not_implemented()
    over_ordered = not_implemented()
    map_batches = not_implemented()
    rolling_expr = not_implemented()
