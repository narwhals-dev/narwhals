from __future__ import annotations

import datetime as dt
from functools import reduce
from typing import TYPE_CHECKING, Any, Literal, cast, overload

import pyarrow as pa  # ignore-banned-import
import pyarrow.compute as pc  # ignore-banned-import

from narwhals._arrow.utils import narwhals_to_native_dtype
from narwhals._plan._guards import is_tuple_of
from narwhals._plan.arrow import functions as fn
from narwhals._plan.compliant.namespace import EagerNamespace
from narwhals._plan.expressions.literal import is_literal_scalar
from narwhals._utils import Implementation, Version
from narwhals.exceptions import InvalidOperationError

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence

    from narwhals._arrow.typing import ChunkedArrayAny
    from narwhals._plan.arrow.dataframe import ArrowDataFrame as Frame
    from narwhals._plan.arrow.expr import ArrowExpr as Expr, ArrowScalar as Scalar
    from narwhals._plan.arrow.series import ArrowSeries as Series
    from narwhals._plan.arrow.typing import ChunkedArray, IntegerScalar
    from narwhals._plan.expressions import expr, functions as F
    from narwhals._plan.expressions.boolean import AllHorizontal, AnyHorizontal
    from narwhals._plan.expressions.expr import FunctionExpr, RangeExpr
    from narwhals._plan.expressions.ranges import DateRange, IntRange
    from narwhals._plan.expressions.strings import ConcatStr
    from narwhals._plan.series import Series as NwSeries
    from narwhals._plan.typing import NonNestedLiteralT
    from narwhals.dtypes import IntegerType
    from narwhals.typing import (
        ClosedInterval,
        ConcatMethod,
        NonNestedLiteral,
        PythonLiteral,
    )


Int64 = Version.MAIN.dtypes.Int64()


class ArrowNamespace(EagerNamespace["Frame", "Series", "Expr", "Scalar"]):
    implementation = Implementation.PYARROW

    def __init__(self, version: Version = Version.MAIN) -> None:
        self._version = version

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

    def col(self, node: expr.Column, frame: Frame, name: str) -> Expr:
        return self._expr.from_native(
            frame.native.column(node.name), name, version=frame.version
        )

    @overload
    def lit(
        self, node: expr.Literal[NonNestedLiteral], frame: Frame, name: str
    ) -> Scalar: ...
    @overload
    def lit(
        self, node: expr.Literal[NwSeries[ChunkedArrayAny]], frame: Frame, name: str
    ) -> Expr: ...
    def lit(
        self,
        node: expr.Literal[NonNestedLiteral] | expr.Literal[NwSeries[ChunkedArrayAny]],
        frame: Frame,
        name: str,
    ) -> Expr | Scalar:
        if is_literal_scalar(node):
            return self._scalar.from_python(
                node.unwrap(), name, dtype=node.dtype, version=frame.version
            )
        nw_ser = node.unwrap()
        return self._expr.from_native(
            nw_ser.to_native(), name or node.name, nw_ser.version
        )

    # NOTE: Update with `ignore_nulls`/`fill_null` behavior once added to each `Function`
    # https://github.com/narwhals-dev/narwhals/pull/2719
    def _horizontal_function(
        self, fn_native: Callable[[Any, Any], Any], /, fill: NonNestedLiteral = None
    ) -> Callable[[FunctionExpr[Any], Frame, str], Expr | Scalar]:
        def func(node: FunctionExpr[Any], frame: Frame, name: str) -> Expr | Scalar:
            it = (self._expr.from_ir(e, frame, name).native for e in node.input)
            if fill is not None:
                it = (pc.fill_null(native, fn.lit(fill)) for native in it)
            result = reduce(fn_native, it)
            if isinstance(result, pa.Scalar):
                return self._scalar.from_native(result, name, self.version)
            return self._expr.from_native(result, name, self.version)

        return func

    def any_horizontal(
        self, node: FunctionExpr[AnyHorizontal], frame: Frame, name: str
    ) -> Expr | Scalar:
        return self._horizontal_function(fn.or_)(node, frame, name)

    def all_horizontal(
        self, node: FunctionExpr[AllHorizontal], frame: Frame, name: str
    ) -> Expr | Scalar:
        return self._horizontal_function(fn.and_)(node, frame, name)

    def sum_horizontal(
        self, node: FunctionExpr[F.SumHorizontal], frame: Frame, name: str
    ) -> Expr | Scalar:
        return self._horizontal_function(fn.add, fill=0)(node, frame, name)

    def min_horizontal(
        self, node: FunctionExpr[F.MinHorizontal], frame: Frame, name: str
    ) -> Expr | Scalar:
        return self._horizontal_function(fn.min_horizontal)(node, frame, name)

    def max_horizontal(
        self, node: FunctionExpr[F.MaxHorizontal], frame: Frame, name: str
    ) -> Expr | Scalar:
        return self._horizontal_function(fn.max_horizontal)(node, frame, name)

    def mean_horizontal(
        self, node: FunctionExpr[F.MeanHorizontal], frame: Frame, name: str
    ) -> Expr | Scalar:
        int64 = pa.int64()
        inputs = [self._expr.from_ir(e, frame, name).native for e in node.input]
        filled = (pc.fill_null(native, fn.lit(0)) for native in inputs)
        # NOTE: `mypy` doesn't like that `add` is overloaded
        sum_not_null = reduce(
            fn.add,  # type: ignore[arg-type]
            (fn.cast(fn.is_not_null(native), int64) for native in inputs),
        )
        result = fn.truediv(reduce(fn.add, filled), sum_not_null)
        if isinstance(result, pa.Scalar):
            return self._scalar.from_native(result, name, self.version)
        return self._expr.from_native(result, name, self.version)

    def concat_str(
        self, node: FunctionExpr[ConcatStr], frame: Frame, name: str
    ) -> Expr | Scalar:
        exprs = (self._expr.from_ir(e, frame, name) for e in node.input)
        aligned = (ser.native for ser in self._expr.align(exprs))
        separator = node.function.separator
        ignore_nulls = node.function.ignore_nulls
        result = fn.concat_str(*aligned, separator=separator, ignore_nulls=ignore_nulls)
        if isinstance(result, pa.Scalar):
            return self._scalar.from_native(result, name, self.version)
        return self._expr.from_native(result, name, self.version)

    def _range_function_inputs(
        self, node: RangeExpr, frame: Frame, valid_type: type[NonNestedLiteralT]
    ) -> tuple[NonNestedLiteralT, NonNestedLiteralT]:
        start_: PythonLiteral
        end_: PythonLiteral
        start, end = node.function.unwrap_input(node)
        if is_literal_scalar(start) and is_literal_scalar(end):
            start_, end_ = start.unwrap(), end.unwrap()
        else:
            scalar_start = self._expr.from_ir(start, frame, "start")
            scalar_end = self._expr.from_ir(end, frame, "end")
            if isinstance(scalar_start, self._scalar) and isinstance(
                scalar_end, self._scalar
            ):
                start_, end_ = scalar_start.to_python(), scalar_end.to_python()
            else:
                msg = (
                    f"All inputs for `{node.function}()` must be scalar or aggregations, but got \n"
                    f"{scalar_start.native!r}\n{scalar_end.native!r}"
                )
                raise InvalidOperationError(msg)
        if isinstance(start_, valid_type) and isinstance(end_, valid_type):
            return start_, end_
        msg = f"All inputs for `{node.function}()` must resolve to {valid_type.__name__}, but got \n{start_!r}\n{end_!r}"
        raise InvalidOperationError(msg)

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
        start, end = self._range_function_inputs(node, frame, int)
        native = self._int_range(start, end, node.function.step, node.function.dtype)
        return self._expr.from_native(native, name, self.version)

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
        return self._series.from_native(native, name, version=self.version)

    def date_range(self, node: RangeExpr[DateRange], frame: Frame, name: str) -> Expr:
        start, end = self._range_function_inputs(node, frame, dt.date)
        func = node.function
        native = fn.date_range(start, end, func.interval, closed=func.closed)
        return self._expr.from_native(native, name, self.version)

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
        return self._series.from_native(native, name, version=self.version)

    @overload
    def concat(self, items: Iterable[Frame], *, how: ConcatMethod) -> Frame: ...
    @overload
    def concat(self, items: Iterable[Series], *, how: Literal["vertical"]) -> Series: ...
    def concat(
        self, items: Iterable[Frame | Series], *, how: ConcatMethod
    ) -> Frame | Series:
        if how == "vertical":
            return self._concat_vertical(items)
        if how == "horizontal":
            return self._concat_horizontal(items)
        it = iter(items)
        first = next(it)
        if self._is_series(first):
            raise TypeError(first)
        dfs = cast("Sequence[Frame]", (first, *it))
        return self._concat_diagonal(dfs)

    def _concat_diagonal(self, items: Iterable[Frame]) -> Frame:
        return self._dataframe.from_native(
            fn.concat_vertical_table(df.native for df in items), self.version
        )

    def _concat_horizontal(self, items: Iterable[Frame | Series]) -> Frame:
        def gen(objs: Iterable[Frame | Series]) -> Iterator[tuple[ChunkedArrayAny, str]]:
            for item in objs:
                if self._is_series(item):
                    yield item.native, item.name
                else:
                    yield from zip(item.native.itercolumns(), item.columns)

        arrays, names = zip(*gen(items))
        native = pa.Table.from_arrays(arrays, list(names))
        return self._dataframe.from_native(native, self.version)

    def _concat_vertical(self, items: Iterable[Frame | Series]) -> Frame | Series:
        collected = items if isinstance(items, tuple) else tuple(items)
        if is_tuple_of(collected, self._series):
            sers = collected
            chunked = fn.concat_vertical_chunked(ser.native for ser in sers)
            return sers[0]._with_native(chunked)
        if is_tuple_of(collected, self._dataframe):
            dfs = collected
            cols_0 = dfs[0].columns
            for i, df in enumerate(dfs[1:], start=1):
                cols_current = df.columns
                if cols_current != cols_0:
                    msg = (
                        "unable to vstack, column names don't match:\n"
                        f"   - dataframe 0: {cols_0}\n"
                        f"   - dataframe {i}: {cols_current}\n"
                    )
                    raise TypeError(msg)
            return df._with_native(fn.concat_vertical_table(df.native for df in dfs))
        raise TypeError(items)
