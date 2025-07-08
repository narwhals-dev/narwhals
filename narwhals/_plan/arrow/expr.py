from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast, overload

import pyarrow as pa  # ignore-banned-import
import pyarrow.compute as pc  # ignore-banned-import

from narwhals._arrow.utils import (
    chunked_array as _chunked_array,
    narwhals_to_native_dtype,
)
from narwhals._plan.arrow.series import ArrowSeries
from narwhals._plan.common import ExprIR, into_dtype
from narwhals._plan.protocols import EagerExpr, EagerScalar, ExprDispatch
from narwhals._utils import Implementation, Version, _StoresNative, not_implemented
from narwhals.exceptions import InvalidOperationError, ShapeError

if TYPE_CHECKING:
    from collections.abc import Iterable

    from typing_extensions import Self, TypeAlias

    from narwhals._arrow.typing import (
        ArrayAny,
        ArrayOrScalar,
        ChunkedArrayAny,
        Incomplete,
    )
    from narwhals._plan import expr
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
    from narwhals.typing import IntoDType, PythonLiteral

NativeScalar: TypeAlias = "pa.Scalar[Any]"

BACKEND_VERSION = Implementation.PYARROW._backend_version()


class ArrowExpr(
    ExprDispatch["ArrowDataFrame", "ArrowExpr | ArrowScalar", "ArrowNamespace"],
    _StoresNative["ChunkedArrayAny"],
    EagerExpr["ArrowDataFrame", ArrowSeries],
):
    _evaluated: ArrowSeries
    _version: Version

    def __narwhals_namespace__(self) -> ArrowNamespace:
        from narwhals._plan.arrow.namespace import ArrowNamespace

        return ArrowNamespace(self._version)

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
    def _with_native(self, result: ChunkedArrayAny, name: str = ..., /) -> Self: ...
    @overload
    def _with_native(self, result: NativeScalar, name: str = ..., /) -> ArrowScalar: ...
    @overload
    def _with_native(
        self, result: ChunkedArrayAny | NativeScalar, name: str = ..., /
    ) -> ArrowScalar | Self: ...
    def _with_native(
        self, result: ChunkedArrayAny | NativeScalar, name: str = "", /
    ) -> ArrowScalar | Self:
        if isinstance(result, pa.Scalar):
            return ArrowScalar.from_native(result, name, version=self.version)
        return super()._with_native(result, name)

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

    def cast(  # type: ignore[override]
        self, node: expr.Cast, frame: ArrowDataFrame, name: str
    ) -> ArrowScalar | Self:
        data_type = narwhals_to_native_dtype(node.dtype, frame.version)
        native = self._dispatch(node.expr, frame, name).native
        return self._with_native(pc.cast(native, data_type), name)

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
            )
        )

    def first(self, node: First, frame: ArrowDataFrame, name: str) -> ArrowScalar:
        prev = self._dispatch_expr(node.expr, frame, name)
        native = prev.native
        result = lit(native[0]) if len(prev) else lit(None, native.type)
        return self._with_native(result, name)

    def last(self, node: Last, frame: ArrowDataFrame, name: str) -> ArrowScalar:
        prev = self._dispatch_expr(node.expr, frame, name)
        native = prev.native
        result = (
            lit(native[height - 1]) if (height := len(prev)) else lit(None, native.type)
        )
        return self._with_native(result, name)

    def arg_min(self, node: ArgMin, frame: ArrowDataFrame, name: str) -> ArrowScalar:
        native = self._dispatch_expr(node.expr, frame, name).native
        result = pc.index(native, pc.min(native))
        return self._with_native(result, name)

    def arg_max(self, node: ArgMax, frame: ArrowDataFrame, name: str) -> ArrowScalar:
        native = self._dispatch_expr(node.expr, frame, name).native
        result: NativeScalar = pc.index(native, pc.max(native))
        return self._with_native(result, name)

    def sum(self, node: Sum, frame: ArrowDataFrame, name: str) -> ArrowScalar:
        result: NativeScalar = pc.sum(
            self._dispatch_expr(node.expr, frame, name).native, min_count=0
        )
        return self._with_native(result, name)

    def n_unique(self, node: NUnique, frame: ArrowDataFrame, name: str) -> ArrowScalar:
        result = pc.count(self._dispatch_expr(node.expr, frame, name).native, mode="all")
        return self._with_native(result, name)

    def std(self, node: Std, frame: ArrowDataFrame, name: str) -> ArrowScalar:
        result = pc.stddev(
            self._dispatch_expr(node.expr, frame, name).native, ddof=node.ddof
        )
        return self._with_native(result, name)

    def var(self, node: Var, frame: ArrowDataFrame, name: str) -> ArrowScalar:
        result = pc.variance(
            self._dispatch_expr(node.expr, frame, name).native, ddof=node.ddof
        )
        return self._with_native(result, name)

    def quantile(self, node: Quantile, frame: ArrowDataFrame, name: str) -> ArrowScalar:
        result = pc.quantile(
            self._dispatch_expr(node.expr, frame, name).native,
            q=node.quantile,
            interpolation=node.interpolation,
        )[0]
        return self._with_native(result, name)

    def count(self, node: Count, frame: ArrowDataFrame, name: str) -> ArrowScalar:
        result = pc.count(self._dispatch_expr(node.expr, frame, name).native)
        return self._with_native(result, name)

    def max(self, node: Max, frame: ArrowDataFrame, name: str) -> ArrowScalar:
        result: NativeScalar = pc.max(self._dispatch_expr(node.expr, frame, name).native)
        return self._with_native(result, name)

    def mean(self, node: Mean, frame: ArrowDataFrame, name: str) -> ArrowScalar:
        result = pc.mean(self._dispatch_expr(node.expr, frame, name).native)
        return self._with_native(result, name)

    def median(self, node: Median, frame: ArrowDataFrame, name: str) -> ArrowScalar:
        result = pc.approximate_median(self._dispatch_expr(node.expr, frame, name).native)
        return self._with_native(result, name)

    def min(self, node: Min, frame: ArrowDataFrame, name: str) -> ArrowScalar:
        result: NativeScalar = pc.min(self._dispatch_expr(node.expr, frame, name).native)
        return self._with_native(result, name)


def lit(value: Any, dtype: pa.DataType | None = None) -> NativeScalar:
    # NOTE: Needed for `pyarrow<13`
    if isinstance(value, pa.Scalar):
        return value
    # NOTE: PR that fixed this the overloads was closed
    # https://github.com/zen-xu/pyarrow-stubs/pull/208
    return pa.scalar(value) if dtype is None else pa.scalar(value, dtype)


# NOTE: https://github.com/apache/arrow/issues/21761
# fmt: off
if BACKEND_VERSION >= (13,):
    def array(value: NativeScalar) -> ArrayAny:
        return pa.array([value], value.type)
else:
    def array(value: NativeScalar) -> ArrayAny:
        return cast("ArrayAny", pa.array([value.as_py()], value.type))
# fmt: on


def chunked_array(
    arr: ArrayOrScalar | list[Iterable[Any]], dtype: pa.DataType | None = None, /
) -> ChunkedArrayAny:
    return _chunked_array(array(arr) if isinstance(arr, pa.Scalar) else arr, dtype)


class ArrowScalar(
    ExprDispatch["ArrowDataFrame", "ArrowScalar", "ArrowNamespace"],
    _StoresNative[NativeScalar],
    EagerScalar["ArrowDataFrame", ArrowSeries],
):
    _name: str
    _evaluated: NativeScalar
    _version: Version

    def __narwhals_namespace__(self) -> ArrowNamespace:
        from narwhals._plan.arrow.namespace import ArrowNamespace

        return ArrowNamespace(self._version)

    @property
    def name(self) -> str:
        return self._name

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

    @property
    def native(self) -> NativeScalar:
        return self._evaluated

    def to_series(self) -> ArrowSeries:
        return self.broadcast(1)

    def broadcast(self, length: int) -> ArrowSeries:
        scalar = self.native
        if length == 1:
            chunked = chunked_array(scalar)
        else:
            # NOTE: Same issue as `pa.scalar` overlapping overloads
            # https://github.com/zen-xu/pyarrow-stubs/pull/209
            pa_repeat: Incomplete = pa.repeat
            chunked = chunked_array(pa_repeat(scalar, length))
        return ArrowSeries.from_native(chunked, self.name, version=self.version)

    def cast(self, node: expr.Cast, frame: ArrowDataFrame, name: str) -> ArrowScalar:
        data_type = narwhals_to_native_dtype(node.dtype, frame.version)
        native = self._dispatch(node.expr, frame, name).native
        return self._with_native(pc.cast(native, data_type), name)

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
