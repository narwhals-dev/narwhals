from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, overload

import pyarrow as pa  # ignore-banned-import
import pyarrow.compute as pc  # ignore-banned-import

from narwhals._arrow.utils import narwhals_to_native_dtype
from narwhals._plan.arrow import functions as fn
from narwhals._plan.arrow.functions import lit
from narwhals._plan.arrow.series import ArrowSeries
from narwhals._plan.arrow.typing import NativeScalar, StoresNativeT_co
from narwhals._plan.common import ExprIR, into_dtype
from narwhals._plan.protocols import EagerExpr, EagerScalar, ExprDispatch
from narwhals._utils import Implementation, Version, _StoresNative, not_implemented
from narwhals.exceptions import InvalidOperationError, ShapeError

if TYPE_CHECKING:
    from collections.abc import Callable

    from typing_extensions import Self

    from narwhals._arrow.typing import ChunkedArrayAny, Incomplete
    from narwhals._plan import boolean, expr
    from narwhals._plan.aggregation import (
        ArgMax,
        ArgMin,
        Count,
        First,
        Last,
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
    from narwhals._plan.arrow.dataframe import ArrowDataFrame
    from narwhals._plan.arrow.namespace import ArrowNamespace
    from narwhals._plan.boolean import IsBetween, IsFinite, IsNan, IsNull
    from narwhals._plan.expr import (
        AnonymousExpr,
        BinaryExpr,
        FunctionExpr,
        OrderedWindowExpr,
        RollingExpr,
        Ternary,
        WindowExpr,
    )
    from narwhals._plan.functions import FillNull, Pow
    from narwhals.typing import IntoDType, PythonLiteral


BACKEND_VERSION = Implementation.PYARROW._backend_version()


class _ArrowDispatch(
    ExprDispatch["ArrowDataFrame", StoresNativeT_co, "ArrowNamespace"], Protocol
):
    """Common to `Expr`, `Scalar` + their dependencies."""

    def __narwhals_namespace__(self) -> ArrowNamespace:
        from narwhals._plan.arrow.namespace import ArrowNamespace

        return ArrowNamespace(self.version)

    def _with_native(self, native: Any, name: str, /) -> StoresNativeT_co: ...
    def cast(self, node: expr.Cast, frame: ArrowDataFrame, name: str) -> StoresNativeT_co:
        data_type = narwhals_to_native_dtype(node.dtype, frame.version)
        native = self._dispatch(node.expr, frame, name).native
        return self._with_native(fn.cast(native, data_type), name)

    def pow(
        self, node: FunctionExpr[Pow], frame: ArrowDataFrame, name: str
    ) -> StoresNativeT_co:
        base, exponent = node.function.unwrap_input(node)
        base_ = self._dispatch(base, frame, "base").native
        exponent_ = self._dispatch(exponent, frame, "exponent").native
        return self._with_native(pc.power(base_, exponent_), name)

    def fill_null(
        self, node: FunctionExpr[FillNull], frame: ArrowDataFrame, name: str
    ) -> StoresNativeT_co:
        expr, value = node.function.unwrap_input(node)
        native = self._dispatch(expr, frame, name).native
        value_ = self._dispatch(value, frame, "value").native
        return self._with_native(pc.fill_null(native, value_), name)

    def is_between(
        self, node: FunctionExpr[IsBetween], frame: ArrowDataFrame, name: str
    ) -> StoresNativeT_co:
        expr, lower_bound, upper_bound = node.function.unwrap_input(node)
        native = self._dispatch(expr, frame, name).native
        lower = self._dispatch(lower_bound, frame, "lower").native
        upper = self._dispatch(upper_bound, frame, "upper").native
        result = fn.is_between(native, lower, upper, node.function.closed)
        return self._with_native(result, name)

    def _unary_function(
        self, fn_native: Callable[[Any], Any], /
    ) -> Callable[[FunctionExpr[Any], ArrowDataFrame, str], StoresNativeT_co]:
        def func(
            node: FunctionExpr[Any], frame: ArrowDataFrame, name: str
        ) -> StoresNativeT_co:
            native = self._dispatch(node.input[0], frame, name).native
            return self._with_native(fn_native(native), name)

        return func

    def not_(
        self, node: FunctionExpr[boolean.Not], frame: ArrowDataFrame, name: str
    ) -> StoresNativeT_co:
        return self._unary_function(pc.invert)(node, frame, name)

    def all(
        self, node: FunctionExpr[boolean.All], frame: ArrowDataFrame, name: str
    ) -> StoresNativeT_co:
        return self._unary_function(fn.all_)(node, frame, name)

    def any(
        self, node: FunctionExpr[boolean.Any], frame: ArrowDataFrame, name: str
    ) -> StoresNativeT_co:
        return self._unary_function(fn.any_)(node, frame, name)

    def is_finite(
        self, node: FunctionExpr[IsFinite], frame: ArrowDataFrame, name: str
    ) -> StoresNativeT_co:
        return self._unary_function(fn.is_finite)(node, frame, name)

    def is_nan(
        self, node: FunctionExpr[IsNan], frame: ArrowDataFrame, name: str
    ) -> StoresNativeT_co:
        return self._unary_function(fn.is_nan)(node, frame, name)

    def is_null(
        self, node: FunctionExpr[IsNull], frame: ArrowDataFrame, name: str
    ) -> StoresNativeT_co:
        return self._unary_function(fn.is_null)(node, frame, name)

    def binary_expr(
        self, node: BinaryExpr, frame: ArrowDataFrame, name: str
    ) -> StoresNativeT_co:
        lhs, rhs = (
            self._dispatch(node.left, frame, name),
            self._dispatch(node.right, frame, name),
        )
        result = fn.binary(lhs.native, node.op.__class__, rhs.native)
        return self._with_native(result, name)

    def ternary_expr(
        self, node: Ternary, frame: ArrowDataFrame, name: str
    ) -> StoresNativeT_co:
        when = self._dispatch(node.predicate, frame, name)
        then = self._dispatch(node.truthy, frame, name)
        otherwise = self._dispatch(node.falsy, frame, name)
        result = pc.if_else(when.native, then.native, otherwise.native)
        return self._with_native(result, name)


class ArrowExpr(  # type: ignore[misc]
    _ArrowDispatch["ArrowExpr | ArrowScalar"],
    _StoresNative["ChunkedArrayAny"],
    EagerExpr["ArrowDataFrame", ArrowSeries],
):
    _evaluated: ArrowSeries
    _version: Version

    @property
    def name(self) -> str:
        return self._evaluated.name

    @classmethod
    def from_series(cls, series: ArrowSeries, /) -> Self:
        obj = cls.__new__(cls)
        obj._evaluated = series
        obj._version = series.version
        return obj

    @classmethod
    def from_native(
        cls, native: ChunkedArrayAny, name: str = "", /, version: Version = Version.MAIN
    ) -> Self:
        return cls.from_series(ArrowSeries.from_native(native, name, version=version))

    @overload
    def _with_native(self, result: ChunkedArrayAny, name: str, /) -> Self: ...
    @overload
    def _with_native(self, result: NativeScalar, name: str, /) -> ArrowScalar: ...
    @overload
    def _with_native(
        self, result: ChunkedArrayAny | NativeScalar, name: str, /
    ) -> ArrowScalar | Self: ...
    def _with_native(
        self, result: ChunkedArrayAny | NativeScalar, name: str, /
    ) -> ArrowScalar | Self:
        if isinstance(result, pa.Scalar):
            return ArrowScalar.from_native(result, name, version=self.version)
        return self.from_native(result, name or self.name, self.version)

    def _dispatch_expr(
        self, node: ExprIR, frame: ArrowDataFrame, name: str
    ) -> ArrowSeries:
        """Use instead of `_dispatch` *iff* an operation isn't natively supported on `ChunkedArray`.

        There is no need to broadcast, as they may have a cheaper impl elsewhere (`CompliantScalar` or `ArrowScalar`).

        Mainly for the benefit of a type checker, but the equivalent `ArrowScalar._dispatch_expr` will raise if
        the assumption fails.
        """
        return self._dispatch(node, frame, name).to_series()

    @property
    def native(self) -> ChunkedArrayAny:
        return self._evaluated.native

    def to_series(self) -> ArrowSeries:
        return self._evaluated

    def broadcast(self, length: int, /) -> ArrowSeries:
        if (actual_len := len(self)) != length:
            msg = f"Expected object of length {length}, got {actual_len}."
            raise ShapeError(msg)
        return self._evaluated

    def __len__(self) -> int:
        return len(self._evaluated)

    def sort(self, node: expr.Sort, frame: ArrowDataFrame, name: str) -> ArrowExpr:
        native = self._dispatch_expr(node.expr, frame, name).native
        sorted_indices = pc.array_sort_indices(native, options=node.options.to_arrow())
        return self._with_native(native.take(sorted_indices), name)

    def sort_by(self, node: expr.SortBy, frame: ArrowDataFrame, name: str) -> ArrowExpr:
        series = self._dispatch_expr(node.expr, frame, name)
        by = (
            self._dispatch_expr(e, frame, f"<TEMP>_{idx}")
            for idx, e in enumerate(node.by)
        )
        df = frame.from_series(series, *by)
        names = df.columns[1:]
        indices = pc.sort_indices(df.native, options=node.options.to_arrow(names))
        result: ChunkedArrayAny = df.native.column(0).take(indices)
        return self._with_native(result, name)

    def filter(self, node: expr.Filter, frame: ArrowDataFrame, name: str) -> ArrowExpr:
        return self._with_native(
            self._dispatch_expr(node.expr, frame, name).native.filter(
                self._dispatch_expr(node.by, frame, name).native
            ),
            name,
        )

    def first(self, node: First, frame: ArrowDataFrame, name: str) -> ArrowScalar:
        prev = self._dispatch_expr(node.expr, frame, name)
        native = prev.native
        result = native[0] if len(prev) else lit(None, native.type)
        return self._with_native(result, name)

    def last(self, node: Last, frame: ArrowDataFrame, name: str) -> ArrowScalar:
        prev = self._dispatch_expr(node.expr, frame, name)
        native = prev.native
        result = native[height - 1] if (height := len(prev)) else lit(None, native.type)
        return self._with_native(result, name)

    def arg_min(self, node: ArgMin, frame: ArrowDataFrame, name: str) -> ArrowScalar:
        native = self._dispatch_expr(node.expr, frame, name).native
        result = pc.index(native, fn.min_(native))
        return self._with_native(result, name)

    def arg_max(self, node: ArgMax, frame: ArrowDataFrame, name: str) -> ArrowScalar:
        native = self._dispatch_expr(node.expr, frame, name).native
        result: NativeScalar = pc.index(native, fn.max_(native))
        return self._with_native(result, name)

    def sum(self, node: Sum, frame: ArrowDataFrame, name: str) -> ArrowScalar:
        result = fn.sum_(self._dispatch_expr(node.expr, frame, name).native)
        return self._with_native(result, name)

    def n_unique(self, node: NUnique, frame: ArrowDataFrame, name: str) -> ArrowScalar:
        result = fn.n_unique(self._dispatch_expr(node.expr, frame, name).native)
        return self._with_native(result, name)

    def std(self, node: Std, frame: ArrowDataFrame, name: str) -> ArrowScalar:
        result = fn.std(
            self._dispatch_expr(node.expr, frame, name).native, ddof=node.ddof
        )
        return self._with_native(result, name)

    def var(self, node: Var, frame: ArrowDataFrame, name: str) -> ArrowScalar:
        result = fn.var(
            self._dispatch_expr(node.expr, frame, name).native, ddof=node.ddof
        )
        return self._with_native(result, name)

    def quantile(self, node: Quantile, frame: ArrowDataFrame, name: str) -> ArrowScalar:
        result = fn.quantile(
            self._dispatch_expr(node.expr, frame, name).native,
            q=node.quantile,
            interpolation=node.interpolation,
        )[0]
        return self._with_native(result, name)

    def count(self, node: Count, frame: ArrowDataFrame, name: str) -> ArrowScalar:
        result = fn.count(self._dispatch_expr(node.expr, frame, name).native)
        return self._with_native(result, name)

    def max(self, node: Max, frame: ArrowDataFrame, name: str) -> ArrowScalar:
        result: NativeScalar = fn.max_(self._dispatch_expr(node.expr, frame, name).native)
        return self._with_native(result, name)

    def mean(self, node: Mean, frame: ArrowDataFrame, name: str) -> ArrowScalar:
        result = fn.mean(self._dispatch_expr(node.expr, frame, name).native)
        return self._with_native(result, name)

    def median(self, node: Median, frame: ArrowDataFrame, name: str) -> ArrowScalar:
        result = fn.median(self._dispatch_expr(node.expr, frame, name).native)
        return self._with_native(result, name)

    def min(self, node: Min, frame: ArrowDataFrame, name: str) -> ArrowScalar:
        result: NativeScalar = fn.min_(self._dispatch_expr(node.expr, frame, name).native)
        return self._with_native(result, name)

    # TODO @dangotbanned: top-level, complex-ish nodes
    # - All are fairly complex
    # - `over`/`_ordered` (with partitions) requires `group_by`, `join`
    # - `over_ordered` alone should be possible w/ the current API
    # - `map_batches` is defined in `EagerExpr`, might be simpler here than on main
    # - `rolling_expr` has 4 variants

    def over(self, node: WindowExpr, frame: ArrowDataFrame, name: str) -> Self:
        raise NotImplementedError

    def over_ordered(
        self, node: OrderedWindowExpr, frame: ArrowDataFrame, name: str
    ) -> Self:
        raise NotImplementedError

    def map_batches(self, node: AnonymousExpr, frame: ArrowDataFrame, name: str) -> Self:
        raise NotImplementedError

    def rolling_expr(self, node: RollingExpr, frame: ArrowDataFrame, name: str) -> Self:
        raise NotImplementedError


class ArrowScalar(
    _ArrowDispatch["ArrowScalar"],
    _StoresNative[NativeScalar],
    EagerScalar["ArrowDataFrame", ArrowSeries],
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
        if dtype:
            dtype = into_dtype(dtype)
            if not isinstance(dtype, version.dtypes.Unknown):
                dtype_pa = narwhals_to_native_dtype(dtype, version)
        return cls.from_native(lit(value, dtype_pa), name, version)

    @classmethod
    def from_series(cls, series: ArrowSeries) -> Self:
        if len(series) == 1:
            return cls.from_native(series.native[0], series.name, series.version)
        elif len(series) == 0:
            return cls.from_python(
                None, series.name, dtype=series.dtype, version=series.version
            )
        else:
            msg = f"Too long {len(series)!r}"
            raise InvalidOperationError(msg)

    def _dispatch_expr(
        self, node: ExprIR, frame: ArrowDataFrame, name: str
    ) -> ArrowSeries:
        msg = f"Expected unreachable, but hit at: {node!r}"
        raise InvalidOperationError(msg)

    def _with_native(self, native: Any, name: str, /) -> Self:
        return self.from_native(native, name or self.name, self.version)

    @property
    def native(self) -> NativeScalar:
        return self._evaluated

    def to_series(self) -> ArrowSeries:
        return self.broadcast(1)

    def to_python(self) -> PythonLiteral:
        return self.native.as_py()  # type: ignore[no-any-return]

    def broadcast(self, length: int) -> ArrowSeries:
        scalar = self.native
        if length == 1:
            chunked = fn.chunked_array(scalar)
        else:
            # NOTE: Same issue as `pa.scalar` overlapping overloads
            # https://github.com/zen-xu/pyarrow-stubs/pull/209
            pa_repeat: Incomplete = pa.repeat
            chunked = fn.chunked_array(pa_repeat(scalar, length))
        return ArrowSeries.from_native(chunked, self.name, version=self.version)

    def arg_min(self, node: ArgMin, frame: ArrowDataFrame, name: str) -> ArrowScalar:
        return self._with_native(pa.scalar(0), name)

    def arg_max(self, node: ArgMax, frame: ArrowDataFrame, name: str) -> ArrowScalar:
        return self._with_native(pa.scalar(0), name)

    def n_unique(self, node: NUnique, frame: ArrowDataFrame, name: str) -> ArrowScalar:
        return self._with_native(pa.scalar(1), name)

    def std(self, node: Std, frame: ArrowDataFrame, name: str) -> ArrowScalar:
        return self._with_native(pa.scalar(None, pa.null()), name)

    def var(self, node: Var, frame: ArrowDataFrame, name: str) -> ArrowScalar:
        return self._with_native(pa.scalar(None, pa.null()), name)

    def count(self, node: Count, frame: ArrowDataFrame, name: str) -> ArrowScalar:
        native = self._dispatch(node.expr, frame, name).native
        return self._with_native(pa.scalar(1 if native.is_valid else 0), name)

    filter = not_implemented()
    over = not_implemented()
    over_ordered = not_implemented()
    map_batches = not_implemented()
    rolling_expr = not_implemented()
