from __future__ import annotations

import datetime as dt
from collections.abc import Collection
from functools import reduce
from typing import TYPE_CHECKING, Any, Literal, overload

import pyarrow as pa  # ignore-banned-import

from narwhals._arrow.utils import narwhals_to_native_dtype
from narwhals._plan import expressions as ir
from narwhals._plan.arrow import functions as fn, io
from narwhals._plan.common import todo
from narwhals._plan.compliant.namespace import EagerNamespace
from narwhals._plan.compliant.translate import FromDict, FromIterable
from narwhals._utils import Implementation, Version
from narwhals.exceptions import InvalidOperationError
from narwhals.schema import Schema

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from typing_extensions import TypeAlias

    from narwhals._plan._dispatch import BoundMethod
    from narwhals._plan.arrow.dataframe import ArrowDataFrame as Frame
    from narwhals._plan.arrow.expr import ArrowExpr as Expr, ArrowScalar as Scalar
    from narwhals._plan.arrow.series import ArrowSeries as Series
    from narwhals._plan.arrow.typing import (
        BinaryFunction,
        ChunkedArray,
        ChunkedArrayAny,
        CompliantDataFrame,
        CompliantSeries,
        IntegerScalar,
        IOSource,
        VariadicFunction,
    )
    from narwhals._plan.expressions import functions as F
    from narwhals._plan.expressions.boolean import AllHorizontal, AnyHorizontal
    from narwhals._plan.expressions.expr import FunctionExpr as FExpr, RangeExpr
    from narwhals._plan.expressions.ranges import DateRange, IntRange, LinearSpace
    from narwhals._plan.expressions.strings import ConcatStr
    from narwhals._plan.typing import NonNestedLiteralT
    from narwhals.dtypes import IntegerType
    from narwhals.typing import (
        ClosedInterval,
        FileSource,
        IntoDType,
        IntoSchema,
        NonNestedLiteral,
        PythonLiteral,
    )

    Wrapper: TypeAlias = BoundMethod[FExpr[Any], Frame, Expr | Scalar]


Int64 = Version.MAIN.dtypes.Int64()


class ArrowNamespace(
    FromIterable["ChunkedArrayAny"],
    FromDict["pa.Table", "ChunkedArrayAny"],
    EagerNamespace["Frame", "Series", "Expr", "Scalar", "pa.Table", "ChunkedArrayAny"],
):
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

    def from_dict(
        self,
        data: Mapping[str, Any],
        /,
        *,
        schema: IntoSchema | None = None,
        version: Version = Version.MAIN,
    ) -> Frame:
        return self._dataframe.from_dict(data, schema=schema, version=version)

    def from_iterable(
        self,
        data: Iterable[Any],
        *,
        name: str = "",
        dtype: IntoDType | None = None,
        version: Version = Version.MAIN,
    ) -> Series:
        return self._series.from_iterable(data, name=name, dtype=dtype, version=version)

    def col(self, node: ir.Column, frame: Frame, name: str) -> Expr:
        return self._expr.from_native(
            frame.native.column(node.name), name, version=frame.version
        )

    def lit(self, node: ir.Lit[NonNestedLiteral], frame: Frame, name: str) -> Scalar:
        return self._scalar.from_python(
            node.value, name, dtype=node.dtype, version=frame.version
        )

    def lit_series(
        self, node: ir.LitSeries[ChunkedArrayAny], frame: Frame, name: str
    ) -> Expr:
        return self._expr.from_native(node.native, name or node.name, node.version)

    @overload
    def _horizontal(
        self, function: BinaryFunction, /, fill: NonNestedLiteral = None
    ) -> Wrapper: ...
    @overload
    def _horizontal(
        self, function: VariadicFunction, /, *, variadic: Literal[True]
    ) -> Wrapper: ...
    def _horizontal(
        self,
        function: BinaryFunction | VariadicFunction,
        /,
        fill: NonNestedLiteral = None,
        *,
        variadic: bool = False,
    ) -> Wrapper:
        """Generate a horizontal wrapper function.

        Arguments:
            function: Native binary or variadic function.
            fill: Fill value to use when nulls should *not* be ignored.
            variadic: If False (default), perform a binary reduction.
                Otherwise, assume we can unpack directly into `function`.
        """

        def func(node: FExpr[Any], frame: Frame, name: str) -> Expr | Scalar:
            it = (self._expr.from_ir(e, frame, name).native for e in node.input)
            if fill is not None:
                it = (fn.fill_null(native, fill) for native in it)
            result = function(*it) if variadic else reduce(function, it)
            if isinstance(result, pa.Scalar):
                return self._scalar.from_native(result, name, self.version)
            return self._expr.from_native(result, name, self.version)

        return func

    def coalesce(self, node: FExpr[F.Coalesce], frame: Frame, name: str) -> Expr | Scalar:
        return self._horizontal(fn.coalesce, variadic=True)(node, frame, name)

    def any_horizontal(
        self, node: FExpr[AnyHorizontal], frame: Frame, name: str
    ) -> Expr | Scalar:
        fill = False if node.function.ignore_nulls else None
        return self._horizontal(fn.or_, fill)(node, frame, name)

    def all_horizontal(
        self, node: FExpr[AllHorizontal], frame: Frame, name: str
    ) -> Expr | Scalar:
        fill = True if node.function.ignore_nulls else None
        return self._horizontal(fn.and_, fill)(node, frame, name)

    def sum_horizontal(
        self, node: FExpr[F.SumHorizontal], frame: Frame, name: str
    ) -> Expr | Scalar:
        return self._horizontal(fn.add, fill=0)(node, frame, name)

    def min_horizontal(
        self, node: FExpr[F.MinHorizontal], frame: Frame, name: str
    ) -> Expr | Scalar:
        return self._horizontal(fn.min_horizontal, variadic=True)(node, frame, name)

    def max_horizontal(
        self, node: FExpr[F.MaxHorizontal], frame: Frame, name: str
    ) -> Expr | Scalar:
        return self._horizontal(fn.max_horizontal, variadic=True)(node, frame, name)

    def mean_horizontal(
        self, node: FExpr[F.MeanHorizontal], frame: Frame, name: str
    ) -> Expr | Scalar:
        int64 = pa.int64()
        inputs = [self._expr.from_ir(e, frame, name).native for e in node.input]
        filled = (fn.fill_null(native, 0) for native in inputs)
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
        self, node: FExpr[ConcatStr], frame: Frame, name: str
    ) -> Expr | Scalar:
        exprs = (self._expr.from_ir(e, frame, name) for e in node.input)
        aligned = (ser.native for ser in self._expr.align(exprs))
        separator = node.function.separator
        ignore_nulls = node.function.ignore_nulls
        result = fn.str.concat_str(
            *aligned, separator=separator, ignore_nulls=ignore_nulls
        )
        if isinstance(result, pa.Scalar):
            return self._scalar.from_native(result, name, self.version)
        return self._expr.from_native(result, name, self.version)

    # TODO @dangotbanned: Refactor alongside `nwp.functions._ensure_range_scalar`
    # Consider returning the supertype of inputs
    def _range_function_inputs(
        self,
        node: RangeExpr,
        frame: Frame,
        valid_type: type[NonNestedLiteralT] | tuple[type[NonNestedLiteralT], ...],
    ) -> tuple[NonNestedLiteralT, NonNestedLiteralT]:
        start_: PythonLiteral
        end_: PythonLiteral
        start, end = node.function.unwrap_input(node)
        if isinstance(start, ir.Lit) and isinstance(end, ir.Lit):
            start_, end_ = start.value, end.value
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
            return start_, end_  # type: ignore[return-value]
        valid_types = (valid_type,) if not isinstance(valid_type, tuple) else valid_type
        tp_names = " | ".join(tp.__name__ for tp in valid_types)
        msg = f"All inputs for `{node.function}()` must resolve to {tp_names}, but got \n{start_!r}\n{end_!r}"
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

    def linear_space(self, node: RangeExpr[LinearSpace], frame: Frame, name: str) -> Expr:
        start, end = self._range_function_inputs(node, frame, (int, float))
        func = node.function
        native = fn.linear_space(start, end, func.num_samples, closed=func.closed)
        return self._expr.from_native(native, name, self.version)

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
        return self._series.from_native(native, name, version=self.version)

    def concat_df_vertical(self, dfs: Iterable[CompliantDataFrame]) -> Frame:
        dfs = dfs if isinstance(dfs, tuple) else tuple(dfs)
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
        result = fn.concat_tables(df.native for df in dfs)
        return self._dataframe.from_native(result, self.version)

    def concat_df_diagonal(self, dfs: Iterable[CompliantDataFrame]) -> Frame:
        return self._dataframe.from_native(
            fn.concat_tables((df.native for df in dfs), "default"), self.version
        )

    def concat_df_horizontal(self, dfs: Iterable[CompliantDataFrame]) -> Frame:
        return self._dataframe.from_native(
            fn.concat_tables_horizontal(df.native for df in dfs), self.version
        )

    def concat_series(self, series: Iterable[CompliantSeries]) -> Series:
        series = series if isinstance(series, tuple) else tuple(series)
        result = fn.concat_vertical(ser.native for ser in series)
        return self._series.from_native(result, series[0].name, version=self.version)

    def concat_series_horizontal(self, series: Iterable[CompliantSeries], /) -> Frame:
        """Used for `ArrowExpr.sort_by`, seems like only pandas needs `stack_horizontal`?"""
        if isinstance(series, Collection):
            arrays, names = [s.native for s in series], [s.name for s in series]
        else:
            arrays, names = [], []
            for s in series:
                arrays.append(s.native)
                names.append(s.name)
        result = fn.concat_horizontal(arrays, names)
        return self._dataframe.from_native(result, self.version)

    def read_csv(self, source: FileSource, /, **kwds: Any) -> Frame:
        native = io.read_csv(source, **kwds)
        return self._dataframe.from_native(native, version=self.version)

    def read_parquet(self, source: IOSource, /, **kwds: Any) -> Frame:
        native = io.read_parquet(source, **kwds)
        return self._dataframe.from_native(native, version=self.version)

    def read_csv_schema(self, source: FileSource, /, **kwds: Any) -> Schema:
        return Schema.from_arrow(io.read_csv_schema(source, **kwds))

    def read_parquet_schema(self, source: IOSource, /) -> Schema:
        return Schema.from_arrow(io.read_parquet_schema(source))

    scan_csv = todo()
    scan_parquet = todo()
