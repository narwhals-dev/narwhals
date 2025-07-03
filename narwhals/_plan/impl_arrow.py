"""Translating `ExprIR` nodes for pyarrow.

Acting like a trimmed down, native-only `CompliantExpr`, `CompliantSeries`, etc.
"""

from __future__ import annotations

import typing as t
from collections.abc import Sequence, Sized

# ruff: noqa: ARG001
from functools import singledispatch
from itertools import chain, repeat

from narwhals._plan import aggregation as agg, expr
from narwhals._plan.common import into_dtype
from narwhals._plan.contexts import ExprContext
from narwhals._plan.dummy import DummyCompliantFrame, DummyCompliantSeries
from narwhals._plan.expr_expansion import into_named_irs, prepare_projection
from narwhals._plan.expr_parsing import parse_into_seq_of_expr_ir
from narwhals._plan.literal import is_literal_scalar
from narwhals._typing_compat import TypeVar
from narwhals._utils import Version
from narwhals.exceptions import InvalidOperationError, ShapeError

if t.TYPE_CHECKING:
    import pyarrow as pa
    from typing_extensions import Self, TypeAlias, TypeIs

    from narwhals._arrow.typing import (  # type: ignore[attr-defined]
        Incomplete,
        Order,
        ScalarAny,
    )
    from narwhals._plan.common import ExprIR, NamedIR
    from narwhals._plan.dummy import DummySeries
    from narwhals._plan.typing import IntoExpr
    from narwhals.dtypes import DType
    from narwhals.schema import Schema
    from narwhals.typing import IntoDType, NonNestedLiteral, PythonLiteral


NativeFrame: TypeAlias = "pa.Table"
NativeSeries: TypeAlias = "pa.ChunkedArray[t.Any]"

UnaryFn: TypeAlias = "t.Callable[[NativeSeries], ScalarAny]"
SeriesT = TypeVar("SeriesT")


def is_series(obj: t.Any) -> TypeIs[ArrowSeries]:
    return isinstance(obj, ArrowSeries)


class ArrowDataFrame(DummyCompliantFrame[NativeFrame, NativeSeries]):
    @property
    def _series(self) -> type[ArrowSeries]:
        return ArrowSeries

    @property
    def columns(self) -> list[str]:
        return self.native.column_names

    @property
    def schema(self) -> dict[str, DType]:
        from narwhals._arrow.utils import native_to_narwhals_dtype

        schema = self.native.schema
        return {
            name: native_to_narwhals_dtype(dtype, self._version)
            for name, dtype in zip(schema.names, schema.types)
        }

    def __len__(self) -> int:
        return len(self.native)

    @classmethod
    def from_series(
        cls, series: t.Iterable[ArrowSeries] | ArrowSeries, *more_series: ArrowSeries
    ) -> Self:
        lhs = (series,) if is_series(series) else series
        it = chain(lhs, more_series) if more_series else lhs
        return cls.from_dict({s.name: s.native for s in it})

    @classmethod
    def from_dict(
        cls,
        data: t.Mapping[str, t.Any],
        /,
        *,
        schema: t.Mapping[str, DType] | Schema | None = None,
    ) -> Self:
        import pyarrow as pa

        from narwhals.schema import Schema

        pa_schema = Schema(schema).to_arrow() if schema is not None else schema
        native = pa.Table.from_pydict(data, schema=pa_schema)
        return cls.from_native(native, version=Version.MAIN)

    def iter_columns(self) -> t.Iterator[ArrowSeries]:
        for name, series in zip(self.columns, self.native.itercolumns()):
            yield ArrowSeries.from_native(series, name, version=self.version)

    @t.overload
    def to_dict(self, *, as_series: t.Literal[True]) -> dict[str, ArrowSeries]: ...

    @t.overload
    def to_dict(self, *, as_series: t.Literal[False]) -> dict[str, list[t.Any]]: ...

    def to_dict(
        self, *, as_series: bool
    ) -> dict[str, ArrowSeries] | dict[str, list[t.Any]]:
        it = self.iter_columns()
        if as_series:
            return {ser.name: ser for ser in it}
        return {ser.name: ser.to_list() for ser in it}

    def _evaluate_irs(
        self, nodes: t.Iterable[NamedIR[ExprIR]], /
    ) -> t.Iterator[ArrowSeries]:
        for node in nodes:
            yield self._series.from_native(
                _evaluate_inner(node.expr, self), node.name, version=self.version
            )

    def select(
        self, *exprs: IntoExpr | t.Iterable[IntoExpr], **named_exprs: t.Any
    ) -> Self:
        irs, schema_frozen, output_names = prepare_projection(
            parse_into_seq_of_expr_ir(*exprs, **named_exprs), self.schema
        )
        named_irs = into_named_irs(irs, output_names)
        named_irs, schema_projected = schema_frozen.project(named_irs, ExprContext.SELECT)
        return self.from_series(self._evaluate_irs(named_irs))


class ArrowSeries(DummyCompliantSeries[NativeSeries]):
    def to_list(self) -> list[t.Any]:
        return self.native.to_pylist()


class SupportsBroadcast(Sized, t.Protocol[SeriesT]):
    """Minimal broadcasting for `Expr` results."""

    @classmethod
    def from_series(cls, series: SeriesT, /) -> Self: ...
    def to_series(self) -> SeriesT: ...
    def broadcast(self, length: int, /) -> SeriesT: ...
    @classmethod
    def align(cls, *exprs: SupportsBroadcast[SeriesT]) -> Sequence[SeriesT]:
        lengths = [len(e) for e in exprs]
        max_length = max(lengths)
        fast_path = all(len_ == max_length for len_ in lengths)
        if fast_path:
            return [e.to_series() for e in exprs]
        return [e.broadcast(max_length) for e in exprs]


# NOTE: General expression result
# Mostly elementwise
class ArrowExpr(SupportsBroadcast[ArrowSeries]):
    _series: ArrowSeries

    @classmethod
    def from_series(cls, series: ArrowSeries) -> Self:
        obj = cls.__new__(cls)
        obj._series = series
        return obj

    def to_series(self) -> ArrowSeries:
        return self._series

    def __len__(self) -> int:
        return len(self._series)

    def broadcast(self, length: int, /) -> ArrowSeries:
        if (actual_len := len(self)) != length:
            msg = f"Expected object of length {length}, got {actual_len}."
            raise ShapeError(msg)
        return self._series


# NOTE: Aggregation result or scalar
# Should handle broadcasting, without exposing it
class ArrowLiteral(SupportsBroadcast[ArrowSeries]):
    _native_scalar: ScalarAny
    _name: str

    @property
    def name(self) -> str:
        return self._name

    def __len__(self) -> int:
        return 1

    def broadcast(self, length: int, /) -> ArrowSeries:
        import pyarrow as pa

        from narwhals._arrow.utils import chunked_array

        if length == 1:
            chunked = chunked_array([[self._native_scalar]])
        else:
            # NOTE: Same issue as `pa.scalar` overlapping overloads
            # https://github.com/zen-xu/pyarrow-stubs/pull/209
            pa_repeat: Incomplete = pa.repeat
            arr = pa_repeat(self._native_scalar, length)
            chunked = chunked_array(arr)
        return ArrowSeries.from_native(chunked, self.name)

    @classmethod
    def from_series(cls, series: ArrowSeries) -> Self:
        if len(series) == 1:
            return cls.from_scalar(series.native[0], series.name)
        elif len(series) == 0:
            return cls.from_python(None, series.name, dtype=series.dtype)
        else:
            msg = f"Too long {len(series)!r}"
            raise InvalidOperationError(msg)

    def to_series(self) -> ArrowSeries:
        return self.broadcast(1)

    @classmethod
    def from_python(
        cls,
        value: PythonLiteral,
        name: str = "literal",
        /,
        *,
        dtype: IntoDType | None = None,
    ) -> Self:
        import pyarrow as pa

        from narwhals._arrow.utils import narwhals_to_native_dtype

        version = Version.MAIN
        dtype_pa: pa.DataType | None = None
        if dtype:
            dtype = into_dtype(dtype)
            if not isinstance(dtype, version.dtypes.Unknown):
                dtype_pa = narwhals_to_native_dtype(dtype, version)
        # NOTE: PR that fixed this was closed
        # https://github.com/zen-xu/pyarrow-stubs/pull/208
        lit: Incomplete = pa.scalar
        return cls.from_scalar(lit(value, dtype_pa), name)

    @classmethod
    def from_scalar(cls, scalar: ScalarAny, name: str = "literal", /) -> Self:
        obj = cls.__new__(cls)
        obj._native_scalar = scalar
        obj._name = name
        return obj

    @classmethod
    def from_ir(cls, value: expr.Literal[NonNestedLiteral], /) -> Self:
        return cls.from_python(value.unwrap(), value.name)


# NOTE: Should mean we produce 1x CompliantSeries for the entire expression
# Multi-output have already been separated
# No intermediate CompliantSeries need to be created, just assign a name to the final one
@singledispatch
def _evaluate_inner(node: ExprIR, frame: ArrowDataFrame) -> NativeSeries:
    raise NotImplementedError(type(node))


@_evaluate_inner.register(expr.Column)
def col(node: expr.Column, frame: ArrowDataFrame) -> NativeSeries:
    return frame.native.column(node.name)


# NOTE: Using a very naïve approach to broadcasting **for now**
# - We already have something that works in main
# - Another approach would be to keep everything wrapped (or aggregated into)  `expr.Literal`
def _lit_native(value: PythonLiteral | ScalarAny, frame: ArrowDataFrame) -> NativeSeries:
    """Will need to support returning a native scalar as well."""
    import pyarrow as pa

    from narwhals._arrow.utils import chunked_array

    lit: t.Any = pa.scalar
    scalar: t.Any = value if isinstance(value, pa.Scalar) else lit(value)
    array = pa.repeat(scalar, len(frame))
    return chunked_array(array)


@_evaluate_inner.register(expr.Literal)
def lit_(
    node: expr.Literal[NonNestedLiteral] | expr.Literal[DummySeries[NativeSeries]],
    frame: ArrowDataFrame,
) -> NativeSeries:
    if is_literal_scalar(node):
        return _lit_native(node.unwrap(), frame)
    return node.unwrap().to_native()


@_evaluate_inner.register(expr.Cast)
def cast_(node: expr.Cast, frame: ArrowDataFrame) -> NativeSeries:
    from narwhals._arrow.utils import narwhals_to_native_dtype

    data_type = narwhals_to_native_dtype(node.dtype, frame.version)
    return _evaluate_inner(node.expr, frame).cast(data_type)


@_evaluate_inner.register(expr.Sort)
def sort(node: expr.Sort, frame: ArrowDataFrame) -> NativeSeries:
    import pyarrow.compute as pc

    native = _evaluate_inner(node.expr, frame)
    sorted_indices = pc.array_sort_indices(native, options=node.options.to_arrow())
    return native.take(sorted_indices)


@_evaluate_inner.register(expr.SortBy)
def sort_by(node: expr.SortBy, frame: ArrowDataFrame) -> NativeSeries:
    opts = node.options
    if len(opts.nulls_last) != 1:
        msg = f"pyarrow doesn't support multiple values for `nulls_last`, got: {opts.nulls_last!r}"
        raise NotImplementedError(msg)
    placement = "at_end" if opts.nulls_last[0] else "at_start"
    from_native = ArrowSeries.from_native
    by = (
        from_native(_evaluate_inner(e, frame), str(idx)) for idx, e in enumerate(node.by)
    )
    df = frame.from_series(from_native(_evaluate_inner(node.expr, frame), "<TEMP>"), *by)
    names = df.columns[1:]
    if len(opts.descending) == 1:
        descending: t.Iterable[bool] = repeat(opts.descending[0], len(names))
    else:
        descending = opts.descending
    sorting: list[tuple[str, Order]] = [
        (key, "descending" if desc else "ascending")
        for key, desc in zip(names, descending)
    ]
    return df.native.sort_by(sorting, null_placement=placement).column(0)


@_evaluate_inner.register(expr.Filter)
def filter_(node: expr.Filter, frame: ArrowDataFrame) -> NativeSeries:
    return _evaluate_inner(node.expr, frame).filter(_evaluate_inner(node.by, frame))


@_evaluate_inner.register(expr.Len)
def len_(node: expr.Len, frame: ArrowDataFrame) -> NativeSeries:
    return _lit_native(len(frame), frame)


@_evaluate_inner.register(expr.Ternary)
def ternary(node: expr.Ternary, frame: ArrowDataFrame) -> NativeSeries:
    raise NotImplementedError(type(node))


@_evaluate_inner.register(agg.Last)
@_evaluate_inner.register(agg.First)
def agg_first_last(node: agg.First | agg.Last, frame: ArrowDataFrame) -> NativeSeries:
    native = _evaluate_inner(node.expr, frame)
    if height := len(native):
        result = native[height - 1 if isinstance(node, agg.Last) else 0]
    else:
        result = None
    return _lit_native(result, frame)


@_evaluate_inner.register(agg.ArgMax)
@_evaluate_inner.register(agg.ArgMin)
def agg_arg_min_max(node: agg.ArgMin | agg.ArgMax, frame: ArrowDataFrame) -> NativeSeries:
    import pyarrow.compute as pc

    native = _evaluate_inner(node.expr, frame)
    fn = pc.min if isinstance(node, agg.ArgMin) else pc.max
    result = pc.index(native, fn(native))
    return _lit_native(result, frame)


@_evaluate_inner.register(agg.Sum)
def agg_sum(node: agg.Sum, frame: ArrowDataFrame) -> NativeSeries:
    import pyarrow.compute as pc

    result = pc.sum(_evaluate_inner(node.expr, frame), min_count=0)
    return _lit_native(result, frame)


@_evaluate_inner.register(agg.NUnique)
def agg_n_unique(node: agg.NUnique, frame: ArrowDataFrame) -> NativeSeries:
    import pyarrow.compute as pc

    result = pc.count(_evaluate_inner(node.expr, frame).unique(), mode="all")
    return _lit_native(result, frame)


@_evaluate_inner.register(agg.Var)
@_evaluate_inner.register(agg.Std)
def agg_std_var(node: agg.Std | agg.Var, frame: ArrowDataFrame) -> NativeSeries:
    import pyarrow.compute as pc

    fn = pc.stddev if isinstance(node, agg.Std) else pc.variance
    result = fn(_evaluate_inner(node.expr, frame), ddof=node.ddof)
    return _lit_native(result, frame)


@_evaluate_inner.register(agg.Quantile)
def agg_quantile(node: agg.Quantile, frame: ArrowDataFrame) -> NativeSeries:
    import pyarrow.compute as pc

    result = pc.quantile(
        _evaluate_inner(node.expr, frame),
        q=node.quantile,
        interpolation=node.interpolation,
    )[0]
    return _lit_native(result, frame)


@_evaluate_inner.register(expr.Agg)
def agg_expr(node: expr.Agg, frame: ArrowDataFrame) -> NativeSeries:
    import pyarrow.compute as pc

    mapping: dict[type[expr.Agg], UnaryFn] = {
        agg.Count: pc.count,
        agg.Max: pc.max,
        agg.Mean: pc.mean,
        agg.Median: pc.approximate_median,
        agg.Min: pc.min,
    }
    if fn := mapping.get(type(node)):
        result = fn(_evaluate_inner(node.expr, frame))
        return _lit_native(result, frame)
    raise NotImplementedError(type(node))


@_evaluate_inner.register(expr.BinaryExpr)
def binary_expr(node: expr.BinaryExpr, frame: ArrowDataFrame) -> NativeSeries:
    raise NotImplementedError(type(node))


@_evaluate_inner.register(expr.FunctionExpr)
def function_expr(node: expr.FunctionExpr[t.Any], frame: ArrowDataFrame) -> NativeSeries:
    raise NotImplementedError(type(node))


@_evaluate_inner.register(expr.RollingExpr)
def rolling_expr(node: expr.RollingExpr[t.Any], frame: ArrowDataFrame) -> NativeSeries:
    raise NotImplementedError(type(node))


@_evaluate_inner.register(expr.WindowExpr)
def window_expr(node: expr.WindowExpr, frame: ArrowDataFrame) -> NativeSeries:
    raise NotImplementedError(type(node))


@_evaluate_inner.register(expr.AnonymousExpr)
def anonymous_expr(node: expr.AnonymousExpr, frame: ArrowDataFrame) -> NativeSeries:
    raise NotImplementedError(type(node))
