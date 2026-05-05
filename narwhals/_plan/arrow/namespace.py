from __future__ import annotations

from functools import reduce
from typing import TYPE_CHECKING, Any, ClassVar, Literal, overload

import pyarrow as pa

from narwhals._arrow.utils import narwhals_to_native_dtype
from narwhals._plan._version import into_version
from narwhals._plan.arrow import functions as fn, io
from narwhals._plan.common import todo
from narwhals._plan.compliant.namespace import EagerNamespace
from narwhals._plan.exceptions import function_arg_non_scalar_error
from narwhals._utils import Implementation, Version

if TYPE_CHECKING:
    import datetime as dt
    from collections.abc import Callable

    from typing_extensions import TypeAlias

    from narwhals._plan import expressions as ir
    from narwhals._plan.arrow.dataframe import ArrowDataFrame as Frame
    from narwhals._plan.arrow.expr import ArrowExpr as Expr, ArrowScalar as Scalar
    from narwhals._plan.arrow.lazyframe import ArrowLazyFrame as LazyFrame
    from narwhals._plan.arrow.series import ArrowSeries as Series
    from narwhals._plan.arrow.typing import (
        BinaryFunction,
        ChunkedArray,
        ChunkedArrayAny,
        ChunkedOrScalarAny,
        IntegerScalar,
        IOSource,
        VariadicFunction,
    )
    from narwhals._plan.expressions import (
        FunctionExpr as FExpr,
        HorizontalExpr as HExpr,
        RangeExpr,
        functions as F,
    )
    from narwhals._plan.expressions.boolean import AllHorizontal, AnyHorizontal
    from narwhals._plan.expressions.ranges import (
        DateRange,
        IntRange,
        LinearSpace,
        RangeFunction,
    )
    from narwhals._plan.expressions.strings import ConcatStr
    from narwhals._plan.typing import NonNestedLiteralT_co
    from narwhals.dtypes import IntegerType
    from narwhals.schema import Schema
    from narwhals.typing import (
        ClosedInterval,
        FileSource,
        NonNestedLiteral,
        PythonLiteral,
    )

    HWrapper: TypeAlias = Callable[[HExpr[Any], Frame, str], Expr | Scalar]


Int64 = Version.MAIN.dtypes.Int64()


class ArrowNamespace(
    EagerNamespace["Frame", "Series", "Expr", "Scalar", "pa.Table", "ChunkedArrayAny"]
):
    implementation = Implementation.PYARROW
    version: ClassVar[Version] = Version.MAIN

    @property
    def _expr(self) -> type[Expr]:
        from narwhals._plan.arrow.expr import ArrowExpr

        return ArrowExpr

    @property
    def _scalar(self) -> type[Scalar]:
        from narwhals._plan.arrow.expr import ArrowScalar

        return ArrowScalar

    @property
    def _series(self) -> type[Series]:
        from narwhals._plan.arrow.series import ArrowSeries

        return ArrowSeries

    @property
    def _dataframe(self) -> type[Frame]:
        from narwhals._plan.arrow.dataframe import ArrowDataFrame

        return ArrowDataFrame

    @property
    def _lazyframe(self) -> type[LazyFrame]:
        from narwhals._plan.arrow.lazyframe import ArrowLazyFrame

        return ArrowLazyFrame

    def col(self, node: ir.Column, frame: Frame, name: str) -> Expr:
        return self._expr.from_native(frame.native.column(node.name), name)

    def lit(self, node: ir.Lit[PythonLiteral], frame: Frame, name: str) -> Scalar:
        return self._scalar.from_python(node.value, name, dtype=node.dtype)

    def lit_series(
        self, node: ir.LitSeries[ChunkedArrayAny], frame: Frame, name: str
    ) -> Expr:
        return self._expr.from_native(node.native, name or node.name)

    def len(self, node: ir.Len, frame: Frame, name: str) -> Scalar:
        return self._scalar.from_python(len(frame), name or node.name, dtype=None)

    @overload
    def _horizontal(
        self, function: BinaryFunction, /, fill: NonNestedLiteral = None
    ) -> HWrapper: ...
    @overload
    def _horizontal(
        self, function: VariadicFunction, /, *, variadic: Literal[True]
    ) -> HWrapper: ...
    def _horizontal(
        self,
        function: BinaryFunction | VariadicFunction,
        /,
        fill: NonNestedLiteral = None,
        *,
        variadic: bool = False,
    ) -> HWrapper:
        """Generate a horizontal wrapper function.

        Arguments:
            function: Native binary or variadic function.
            fill: Fill value to use when nulls should *not* be ignored.
            variadic: If False (default), perform a binary reduction.
                Otherwise, assume we can unpack directly into `function`.
        """

        def func(node: FExpr[Any], frame: Frame, name: str) -> Expr | Scalar:
            it = (self.from_ir(e, frame, name).native for e in node.input)
            if fill is not None:
                it = (fn.fill_null(native, fill) for native in it)
            result = function(*it) if variadic else reduce(function, it)
            return self._into_expr(result, name)

        return func

    def coalesce(self, node: HExpr[F.Coalesce], frame: Frame, name: str) -> Expr | Scalar:
        return self._horizontal(fn.coalesce, variadic=True)(node, frame, name)

    def any_horizontal(
        self, node: HExpr[AnyHorizontal], frame: Frame, name: str
    ) -> Expr | Scalar:
        fill = False if node.function.ignore_nulls else None
        return self._horizontal(fn.or_, fill)(node, frame, name)

    def all_horizontal(
        self, node: HExpr[AllHorizontal], frame: Frame, name: str
    ) -> Expr | Scalar:
        fill = True if node.function.ignore_nulls else None
        return self._horizontal(fn.and_, fill)(node, frame, name)

    def sum_horizontal(
        self, node: HExpr[F.SumHorizontal], frame: Frame, name: str
    ) -> Expr | Scalar:
        return self._horizontal(fn.add, fill=0)(node, frame, name)

    def min_horizontal(
        self, node: HExpr[F.MinHorizontal], frame: Frame, name: str
    ) -> Expr | Scalar:
        return self._horizontal(fn.min_horizontal, variadic=True)(node, frame, name)

    def max_horizontal(
        self, node: HExpr[F.MaxHorizontal], frame: Frame, name: str
    ) -> Expr | Scalar:
        return self._horizontal(fn.max_horizontal, variadic=True)(node, frame, name)

    def mean_horizontal(
        self, node: HExpr[F.MeanHorizontal], frame: Frame, name: str
    ) -> Expr | Scalar:
        int64 = pa.int64()
        inputs = [self.from_ir(e, frame, name).native for e in node.input]
        filled = (fn.fill_null(native, 0) for native in inputs)
        # NOTE: `mypy` doesn't like that `add` is overloaded
        sum_not_null = reduce(
            fn.add,  # type: ignore[arg-type]
            (fn.cast(fn.is_not_null(native), int64) for native in inputs),
        )
        result = fn.truediv(reduce(fn.add, filled), sum_not_null)
        return self._into_expr(result, name)

    def concat_str(
        self, node: HExpr[ConcatStr], frame: Frame, name: str
    ) -> Expr | Scalar:
        exprs = (self.from_ir(e, frame, name) for e in node.input)
        aligned = (ser.native for ser in self._expr.align(exprs))
        separator = node.function.separator
        ignore_nulls = node.function.ignore_nulls
        result = fn.str.concat_str(
            *aligned, separator=separator, ignore_nulls=ignore_nulls
        )
        return self._into_expr(result, name)

    def _into_expr(self, native: ChunkedOrScalarAny, name: str) -> Expr | Scalar:
        if isinstance(native, pa.Scalar):
            return self._scalar.from_native(native, name)
        return self._expr.from_native(native, name)

    # TODO @dangotbanned: Consider returning the supertype of inputs
    def _range_function_inputs(
        self, node: RangeExpr[RangeFunction[NonNestedLiteralT_co]], frame: Frame
    ) -> tuple[NonNestedLiteralT_co, NonNestedLiteralT_co]:
        func = node.function
        if fastpath := func.try_unwrap_literals(node):
            return fastpath
        _start, _end = node.input
        start = self.from_ir(_start, frame, "")
        end = self.from_ir(_end, frame, "")
        if isinstance(start, self._scalar) and isinstance(end, self._scalar):
            return func.ensure_py_scalars(start.to_python(), end.to_python())
        # TODO @dangotbanned: Add some variant of `self._expr.from_ir` that ensures we got a `ArrowScalar`
        # This should be unreachable, but the typing doesn't know that
        bad = _start if isinstance(start, self._scalar) else _end
        raise function_arg_non_scalar_error(func, bad)

    def _int_range(
        self, start: int, end: int, step: int, dtype: IntegerType, /
    ) -> ChunkedArray[IntegerScalar]:
        if dtype is not Int64:
            pa_dtype = narwhals_to_native_dtype(dtype, self.version)
            if not pa.types.is_integer(pa_dtype):
                raise TypeError(dtype)
            return fn.int_range(start, end, step, dtype=pa_dtype)
        return fn.int_range(start, end, step)

    def int_range(self, node: RangeExpr[IntRange], frame: Frame, name: str) -> Expr:
        start, end = self._range_function_inputs(node, frame)
        native = self._int_range(start, end, node.function.step, node.function.dtype)
        return self._expr.from_native(native, name)

    def int_range_eager(
        self,
        start: int,
        end: int,
        step: int = 1,
        *,
        dtype: IntegerType = Int64,
        name: str = "literal",
    ) -> Series:
        native = self._int_range(start, end, step, dtype)
        return self._series.from_native(native, name)

    def date_range(self, node: RangeExpr[DateRange], frame: Frame, name: str) -> Expr:
        start, end = self._range_function_inputs(node, frame)
        func = node.function
        native = fn.date_range(start, end, func.interval, closed=func.closed)
        return self._expr.from_native(native, name)

    def date_range_eager(
        self,
        start: dt.date,
        end: dt.date,
        interval: int = 1,
        *,
        closed: ClosedInterval = "both",
        name: str = "literal",
    ) -> Series:
        native = fn.date_range(start, end, interval, closed=closed)
        return self._series.from_native(native, name)

    def linear_space(self, node: RangeExpr[LinearSpace], frame: Frame, name: str) -> Expr:
        start, end = self._range_function_inputs(node, frame)
        func = node.function
        native = fn.linear_space(start, end, func.num_samples, closed=func.closed)
        return self._expr.from_native(native, name)

    def linear_space_eager(
        self,
        start: float,
        end: float,
        num_samples: int,
        *,
        closed: ClosedInterval = "both",
        name: str = "literal",
    ) -> Series:
        native = fn.linear_space(start, end, num_samples, closed=closed)
        return self._series.from_native(native, name)

    def read_csv_schema(self, source: FileSource, /, **kwds: Any) -> Schema:
        return into_version(self).schema.from_arrow(io.read_csv_schema(source, **kwds))

    def read_parquet_schema(self, source: IOSource, /) -> Schema:
        return into_version(self).schema.from_arrow(io.read_parquet_schema(source))

    scan_csv = todo()
    scan_parquet = todo()
