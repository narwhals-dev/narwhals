"""Translating `ExprIR` nodes for pyarrow.

Acting like a trimmed down, native-only `CompliantExpr`, `CompliantSeries`, etc.
"""

from __future__ import annotations

import typing as t
from functools import singledispatch
from itertools import chain

from narwhals._plan import expr
from narwhals._plan.contexts import ExprContext
from narwhals._plan.dummy import DummyCompliantFrame, DummyCompliantSeries
from narwhals._plan.expr_expansion import into_named_irs, prepare_projection
from narwhals._plan.expr_parsing import parse_into_seq_of_expr_ir
from narwhals._plan.literal import is_literal_scalar
from narwhals._utils import Version

if t.TYPE_CHECKING:
    import pyarrow as pa
    from typing_extensions import Self, TypeAlias, TypeIs

    from narwhals._plan.common import ExprIR, NamedIR
    from narwhals._plan.dummy import DummySeries
    from narwhals._plan.typing import IntoExpr
    from narwhals.dtypes import DType
    from narwhals.schema import Schema
    from narwhals.typing import NonNestedLiteral


NativeFrame: TypeAlias = "pa.Table"
NativeSeries: TypeAlias = "pa.ChunkedArray[t.Any]"


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


# NOTE: Should mean we produce 1x CompliantSeries for the entire expression
# Multi-output have already been separated
# No intermediate CompliantSeries need to be created, just assign a name to the final one
@singledispatch
def _evaluate_inner(node: ExprIR, frame: ArrowDataFrame) -> NativeSeries:
    raise NotImplementedError(type(node))


@_evaluate_inner.register(expr.Column)
def col(node: expr.Column, frame: ArrowDataFrame) -> NativeSeries:
    return frame.native.column(node.name)


@_evaluate_inner.register(expr.Literal)
def lit_(
    node: expr.Literal[NonNestedLiteral] | expr.Literal[DummySeries[NativeSeries]],
    frame: ArrowDataFrame,
) -> NativeSeries:
    import pyarrow as pa

    if is_literal_scalar(node):
        lit: t.Any = pa.scalar
        array = pa.repeat(lit(node.unwrap()), len(frame))
        return pa.chunked_array([array])
    return node.unwrap().to_native()


@_evaluate_inner.register(expr.Len)
def len_(node: expr.Len, frame: ArrowDataFrame) -> NativeSeries:
    raise NotImplementedError(type(node))


@_evaluate_inner.register(expr.Cast)
def cast_(node: expr.Cast, frame: ArrowDataFrame) -> NativeSeries:
    raise NotImplementedError(type(node))


@_evaluate_inner.register(expr.Ternary)
def ternary(node: expr.Ternary, frame: ArrowDataFrame) -> NativeSeries:
    raise NotImplementedError(type(node))


@_evaluate_inner.register(expr.Agg)
def agg(node: expr.Agg, frame: ArrowDataFrame) -> NativeSeries:
    raise NotImplementedError(type(node))


@_evaluate_inner.register(expr.OrderableAgg)
def orderable_agg(node: expr.OrderableAgg, frame: ArrowDataFrame) -> NativeSeries:
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


@_evaluate_inner.register(expr.Sort)
def sort(node: expr.Sort, frame: ArrowDataFrame) -> NativeSeries:
    raise NotImplementedError(type(node))


@_evaluate_inner.register(expr.SortBy)
def sort_by(node: expr.SortBy, frame: ArrowDataFrame) -> NativeSeries:
    raise NotImplementedError(type(node))


@_evaluate_inner.register(expr.Filter)
def filter_(node: expr.Filter, frame: ArrowDataFrame) -> NativeSeries:
    raise NotImplementedError(type(node))


@_evaluate_inner.register(expr.AnonymousExpr)
def anonymous_expr(node: expr.AnonymousExpr, frame: ArrowDataFrame) -> NativeSeries:
    raise NotImplementedError(type(node))
