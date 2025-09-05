from __future__ import annotations

import typing as t

import pyarrow as pa  # ignore-banned-import
import pyarrow.compute as pc  # ignore-banned-import

from narwhals._arrow.utils import native_to_narwhals_dtype
from narwhals._plan.arrow import functions as fn
from narwhals._plan.arrow.series import ArrowSeries as Series
from narwhals._plan.protocols import EagerDataFrame, namespace
from narwhals._utils import Version

if t.TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

    from typing_extensions import Self

    from narwhals._arrow.typing import ChunkedArrayAny
    from narwhals._plan.arrow.expr import ArrowExpr as Expr, ArrowScalar as Scalar
    from narwhals._plan.arrow.namespace import ArrowNamespace
    from narwhals._plan.common import ExprIR, NamedIR
    from narwhals._plan.dummy import DataFrame as NwDataFrame
    from narwhals._plan.options import SortMultipleOptions
    from narwhals._plan.typing import Seq
    from narwhals.dtypes import DType
    from narwhals.schema import Schema


class ArrowDataFrame(EagerDataFrame[Series, "pa.Table", "ChunkedArrayAny"]):
    def __narwhals_namespace__(self) -> ArrowNamespace:
        from narwhals._plan.arrow.namespace import ArrowNamespace

        return ArrowNamespace(self._version)

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
        return self.native.num_rows

    def to_narwhals(self) -> NwDataFrame[pa.Table, ChunkedArrayAny]:
        from narwhals._plan.dummy import DataFrame

        return DataFrame[pa.Table, "ChunkedArrayAny"]._from_compliant(self)

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

    def iter_columns(self) -> t.Iterator[Series]:
        for name, series in zip(self.columns, self.native.itercolumns()):
            yield Series.from_native(series, name, version=self.version)

    @t.overload
    def to_dict(self, *, as_series: t.Literal[True]) -> dict[str, Series]: ...
    @t.overload
    def to_dict(self, *, as_series: t.Literal[False]) -> dict[str, list[t.Any]]: ...
    @t.overload
    def to_dict(
        self, *, as_series: bool
    ) -> dict[str, Series] | dict[str, list[t.Any]]: ...
    def to_dict(self, *, as_series: bool) -> dict[str, Series] | dict[str, list[t.Any]]:
        it = self.iter_columns()
        if as_series:
            return {ser.name: ser for ser in it}
        return {ser.name: ser.to_list() for ser in it}

    def _evaluate_irs(self, nodes: Iterable[NamedIR[ExprIR]], /) -> Iterator[Series]:
        ns = namespace(self)
        from_named_ir = ns._expr.from_named_ir
        yield from ns._expr.align(from_named_ir(e, self) for e in nodes)

    def sort(self, by: Seq[NamedIR], options: SortMultipleOptions) -> Self:
        df_by = self.select(by)
        indices = pc.sort_indices(df_by.native, options=options.to_arrow(df_by.columns))
        return self._with_native(self.native.take(indices))

    def with_row_index(self, name: str) -> Self:
        return self._with_native(self.native.add_column(0, name, fn.int_range(len(self))))

    def get_column(self, name: str) -> Series:
        chunked = self.native.column(name)
        return Series.from_native(chunked, name, version=self.version)

    def drop(self, columns: Sequence[str]) -> Self:
        to_drop = list(columns)
        return self._with_native(self.native.drop(to_drop))

    # NOTE: Use instead of `with_columns` for trivial cases
    def _with_columns(self, exprs: Iterable[Expr | Scalar], /) -> Self:
        native = self.native
        columns = self.columns
        height = len(self)
        for into_series in exprs:
            name = into_series.name
            chunked = into_series.broadcast(height).native
            if name in columns:
                i = columns.index(name)
                native = native.set_column(i, name, chunked)
            else:
                native = native.append_column(name, chunked)
        return self._with_native(native)
