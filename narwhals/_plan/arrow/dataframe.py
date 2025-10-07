from __future__ import annotations

import operator
from functools import reduce
from itertools import chain
from typing import TYPE_CHECKING, Any, Literal, cast, overload

import pyarrow as pa  # ignore-banned-import
import pyarrow.compute as pc  # ignore-banned-import

from narwhals._arrow.utils import native_to_narwhals_dtype
from narwhals._plan.arrow import acero, functions as fn
from narwhals._plan.arrow.expr import ArrowExpr as Expr, ArrowScalar as Scalar
from narwhals._plan.arrow.group_by import ArrowGroupBy as GroupBy
from narwhals._plan.arrow.series import ArrowSeries as Series
from narwhals._plan.compliant.dataframe import EagerDataFrame
from narwhals._plan.compliant.typing import namespace
from narwhals._plan.expressions import NamedIR
from narwhals._plan.typing import Seq
from narwhals._utils import Implementation, Version, parse_columns_to_drop
from narwhals.schema import Schema

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence

    from typing_extensions import Self

    from narwhals._arrow.typing import ChunkedArrayAny
    from narwhals._plan.arrow.namespace import ArrowNamespace
    from narwhals._plan.expressions import ExprIR, NamedIR
    from narwhals._plan.options import SortMultipleOptions
    from narwhals._plan.typing import NonCrossJoinStrategy, Seq
    from narwhals.dtypes import DType
    from narwhals.typing import IntoSchema


class ArrowDataFrame(EagerDataFrame[Series, "pa.Table", "ChunkedArrayAny"]):
    implementation = Implementation.PYARROW
    _native: pa.Table
    _version: Version

    def __narwhals_namespace__(self) -> ArrowNamespace:
        from narwhals._plan.arrow.namespace import ArrowNamespace

        return ArrowNamespace(self._version)

    @property
    def _group_by(self) -> type[GroupBy]:
        return GroupBy

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

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any], /, *, schema: IntoSchema | None = None
    ) -> Self:
        pa_schema = Schema(schema).to_arrow() if schema is not None else schema
        native = pa.Table.from_pydict(data, schema=pa_schema)
        return cls.from_native(native, version=Version.MAIN)

    def iter_columns(self) -> Iterator[Series]:
        for name, series in zip(self.columns, self.native.itercolumns()):
            yield Series.from_native(series, name, version=self.version)

    @overload
    def to_dict(self, *, as_series: Literal[True]) -> dict[str, Series]: ...
    @overload
    def to_dict(self, *, as_series: Literal[False]) -> dict[str, list[Any]]: ...
    @overload
    def to_dict(self, *, as_series: bool) -> dict[str, Series] | dict[str, list[Any]]: ...
    def to_dict(self, *, as_series: bool) -> dict[str, Series] | dict[str, list[Any]]:
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

    def drop(self, columns: Sequence[str], *, strict: bool = True) -> Self:
        to_drop = parse_columns_to_drop(self, columns, strict=strict)
        return self._with_native(self.native.drop(to_drop))

    def drop_nulls(self, subset: Sequence[str] | None) -> Self:
        if subset is None:
            native = self.native.drop_null()
        else:
            to_drop = reduce(operator.or_, (pc.field(name).is_null() for name in subset))
            native = self.native.filter(~to_drop)
        return self._with_native(native)

    def rename(self, mapping: Mapping[str, str]) -> Self:
        names: dict[str, str] | list[str]
        if fn.BACKEND_VERSION >= (17,):
            names = cast("dict[str, str]", mapping)
        else:  # pragma: no cover
            names = [mapping.get(c, c) for c in self.columns]
        return self._with_native(self.native.rename_columns(names))

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

    def select_names(self, *column_names: str) -> Self:
        return self._with_native(self.native.select(list(column_names)))

    def row(self, index: int) -> tuple[Any, ...]:
        row = self.native.slice(index, 1)
        return tuple(chain.from_iterable(row.to_pydict().values()))

    def join(
        self,
        other: Self,
        *,
        how: NonCrossJoinStrategy,
        left_on: Sequence[str],
        right_on: Sequence[str],
        suffix: str = "_right",
    ) -> Self:
        left, right = self.native, other.native
        result = acero.join_tables(left, right, how, left_on, right_on, suffix=suffix)
        return self._with_native(result)

    def join_cross(self, other: Self, *, suffix: str = "_right") -> Self:
        result = acero.join_cross_tables(self.native, other.native, suffix=suffix)
        return self._with_native(result)

    def filter(self, predicate: NamedIR | Series) -> Self:
        mask: pc.Expression | ChunkedArrayAny
        if not fn.is_series(predicate):
            resolved = Expr.from_named_ir(predicate, self)
            if isinstance(resolved, Expr):
                mask = resolved.broadcast(len(self)).native
            else:
                mask = acero.lit(resolved.native)
        else:
            mask = predicate.native
        return self._with_native(self.native.filter(mask))
