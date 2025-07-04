from __future__ import annotations

import typing as t
from itertools import chain

import pyarrow as pa

from narwhals._arrow.utils import native_to_narwhals_dtype
from narwhals._plan.arrow.expr import ArrowExpr, ArrowLiteral
from narwhals._plan.arrow.series import ArrowSeries
from narwhals._plan.contexts import ExprContext
from narwhals._plan.dummy import DummyCompliantFrame
from narwhals._plan.expr_expansion import into_named_irs, prepare_projection
from narwhals._plan.expr_parsing import parse_into_seq_of_expr_ir
from narwhals._utils import Version

if t.TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from typing_extensions import Self, TypeAlias, TypeIs

    from narwhals._arrow.typing import ChunkedArrayAny, ScalarAny
    from narwhals._plan.common import ExprIR, NamedIR
    from narwhals._plan.typing import IntoExpr
    from narwhals.dtypes import DType
    from narwhals.schema import Schema


UnaryFn: TypeAlias = "t.Callable[[ChunkedArrayAny], ScalarAny]"


def is_series(obj: t.Any) -> TypeIs[ArrowSeries]:
    return isinstance(obj, ArrowSeries)


class ArrowDataFrame(DummyCompliantFrame["pa.Table", "ChunkedArrayAny"]):
    @property
    def _series(self) -> type[ArrowSeries]:
        return ArrowSeries

    @property
    def _expr(self) -> type[ArrowExpr]:
        return ArrowExpr

    @property
    def _lit(self) -> type[ArrowLiteral]:
        return ArrowLiteral

    @property
    def columns(self) -> list[str]:
        return self.native.column_names

    @property
    def schema(self) -> dict[str, DType]:
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

    def _evaluate_irs(self, nodes: Iterable[NamedIR[ExprIR]], /) -> Iterator[ArrowSeries]:
        from narwhals._plan.arrow.evaluate import evaluate

        yield from self._expr.align(evaluate(e, self) for e in nodes)

    def select(
        self, *exprs: IntoExpr | t.Iterable[IntoExpr], **named_exprs: t.Any
    ) -> Self:
        irs, schema_frozen, output_names = prepare_projection(
            parse_into_seq_of_expr_ir(*exprs, **named_exprs), self.schema
        )
        named_irs = into_named_irs(irs, output_names)
        named_irs, schema_projected = schema_frozen.project(named_irs, ExprContext.SELECT)
        return self.from_series(self._evaluate_irs(named_irs))
