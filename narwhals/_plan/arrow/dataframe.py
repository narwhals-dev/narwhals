from __future__ import annotations

import operator
from functools import reduce
from itertools import chain
from typing import TYPE_CHECKING, Any, Literal, cast, overload

import pyarrow as pa  # ignore-banned-import
import pyarrow.compute as pc  # ignore-banned-import

from narwhals._arrow.utils import native_to_narwhals_dtype
from narwhals._plan.arrow import acero, functions as fn
from narwhals._plan.arrow.common import ArrowFrameSeries as FrameSeries
from narwhals._plan.arrow.expr import ArrowExpr as Expr, ArrowScalar as Scalar
from narwhals._plan.arrow.group_by import ArrowGroupBy as GroupBy, partition_by
from narwhals._plan.arrow.series import ArrowSeries as Series
from narwhals._plan.compliant.dataframe import EagerDataFrame
from narwhals._plan.compliant.typing import namespace
from narwhals._plan.exceptions import shape_error
from narwhals._plan.expressions import NamedIR
from narwhals._utils import Version, generate_repr
from narwhals.schema import Schema

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence

    import polars as pl
    from typing_extensions import Self, TypeAlias

    from narwhals._plan.arrow.typing import ChunkedArrayAny, ChunkedOrArrayAny
    from narwhals._plan.compliant.group_by import GroupByResolver
    from narwhals._plan.expressions import ExprIR, NamedIR
    from narwhals._plan.options import ExplodeOptions, SortMultipleOptions
    from narwhals._plan.typing import NonCrossJoinStrategy
    from narwhals.dtypes import DType
    from narwhals.typing import IntoSchema

Incomplete: TypeAlias = Any


class ArrowDataFrame(
    FrameSeries["pa.Table"], EagerDataFrame[Series, "pa.Table", "ChunkedArrayAny"]
):
    def __repr__(self) -> str:
        return generate_repr(f"nw.{type(self).__name__}", self.native.__repr__())

    def _with_native(self, native: pa.Table) -> Self:
        return self.from_native(native, self.version)

    @property
    def _group_by(self) -> type[GroupBy]:
        return GroupBy

    @property
    def shape(self) -> tuple[int, int]:
        return self.native.shape

    def group_by_resolver(self, resolver: GroupByResolver, /) -> GroupBy:
        return self._group_by.from_resolver(self, resolver)

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
        cls,
        data: Mapping[str, Any],
        /,
        *,
        schema: IntoSchema | None = None,
        version: Version = Version.MAIN,
    ) -> Self:
        pa_schema = Schema(schema).to_arrow() if schema is not None else schema
        native = pa.Table.from_pydict(data, schema=pa_schema)
        return cls.from_native(native, version=version)

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

    def to_polars(self) -> pl.DataFrame:
        import polars as pl  # ignore-banned-import
        # NOTE: Recommended in https://github.com/pola-rs/polars/issues/22921#issuecomment-2908506022

        return pl.DataFrame(self.native)

    def _evaluate_irs(
        self, nodes: Iterable[NamedIR[ExprIR]], /, *, length: int | None = None
    ) -> Iterator[Series]:
        expr = namespace(self)._expr
        from_named_ir = expr.from_named_ir
        yield from expr.align((from_named_ir(e, self) for e in nodes), default=length)

    def sort(self, by: Sequence[str], options: SortMultipleOptions | None = None) -> Self:
        return self.gather(fn.sort_indices(self.native, *by, options=options))

    def with_row_index(self, name: str) -> Self:
        return self._with_native(self.native.add_column(0, name, fn.int_range(len(self))))

    def with_row_index_by(
        self,
        name: str,
        order_by: Sequence[str],
        *,
        descending: bool = False,
        nulls_last: bool = False,
    ) -> Self:
        indices = fn.sort_indices(
            self.native, *order_by, nulls_last=nulls_last, descending=descending
        )
        column = fn.unsort_indices(indices)
        return self._with_native(self.native.add_column(0, name, column))

    def to_struct(self, name: str = "") -> Series:
        native = self.native
        if fn.TO_STRUCT_ARRAY_ACCEPTS_EMPTY:
            struct = native.to_struct_array()
        elif fn.HAS_FROM_TO_STRUCT_ARRAY:
            if len(native):
                struct = native.to_struct_array()
            else:
                struct = fn.chunked_array([], pa.struct(native.schema))
        else:
            struct = fn.struct(native.column_names, native.columns)
        return Series.from_native(struct, name, version=self.version)

    def get_column(self, name: str) -> Series:
        chunked = self.native.column(name)
        return Series.from_native(chunked, name, version=self.version)

    def drop(self, columns: Sequence[str]) -> Self:
        return self._with_native(self.native.drop(list(columns)))

    def drop_nulls(self, subset: Sequence[str] | None) -> Self:
        if subset is None:
            native = self.native.drop_null()
        else:
            to_drop = reduce(operator.or_, (pc.field(name).is_null() for name in subset))
            native = self.native.filter(~to_drop)
        return self._with_native(native)

    def explode(self, subset: Sequence[str], options: ExplodeOptions) -> Self:
        builder = fn.ExplodeBuilder.from_options(options)
        if len(subset) == 1:
            return self._with_native(builder.explode_column(self.native, subset[0]))
        return self._with_native(builder.explode_columns(self.native, subset))

    def rename(self, mapping: Mapping[str, str]) -> Self:
        names: dict[str, str] | list[str]
        if fn.BACKEND_VERSION >= (17,):
            names = cast("dict[str, str]", mapping)
        else:  # pragma: no cover
            names = [mapping.get(c, c) for c in self.columns]
        return self._with_native(self.native.rename_columns(names))

    def with_series(self, series: Series) -> Self:
        """Add a new column or replace an existing one.

        Uses similar semantics as `with_columns`, but:
        - for a single named `Series`
        - no broadcasting (use `Scalar.broadcast` instead)
        - no length checking (use `with_series_checked` instead)
        """
        return self._with_native(with_array(self.native, series.name, series.native))

    def with_series_checked(self, series: Series) -> Self:
        expected, actual = len(self), len(series)
        if len(series) != len(self):
            raise shape_error(expected, actual)
        return self.with_series(series)

    def _with_columns(self, exprs: Iterable[Expr | Scalar], /) -> Self:
        height = len(self)
        names_and_columns = ((e.name, e.broadcast(height).native) for e in exprs)
        return self._with_native(with_arrays(self.native, names_and_columns))

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
        right_on: Sequence[str] = (),
        suffix: str = "_right",
    ) -> Self:
        left, right = self.native, other.native
        result = acero.join_tables(left, right, how, left_on, right_on, suffix=suffix)
        return self._with_native(result)

    def join_cross(self, other: Self, *, suffix: str = "_right") -> Self:
        result = acero.join_cross_tables(self.native, other.native, suffix=suffix)
        return self._with_native(result)

    def join_inner(self, other: Self, on: list[str], /) -> Self:
        """Less flexible, but more direct equivalent to join(how="inner", left_on=...)`."""
        return self._with_native(acero.join_inner_tables(self.native, other.native, on))

    def filter(self, predicate: NamedIR) -> Self:
        mask: pc.Expression | ChunkedArrayAny
        resolved = Expr.from_named_ir(predicate, self)
        if isinstance(resolved, Expr):
            mask = resolved.broadcast(len(self)).native
        else:
            mask = acero.lit(resolved.native)
        return self._with_native(self.native.filter(mask))

    def partition_by(self, by: Sequence[str], *, include_key: bool = True) -> list[Self]:
        from_native = self._with_native
        partitions = partition_by(self.native, by, include_key=include_key)
        return [from_native(df) for df in partitions]


def with_array(table: pa.Table, name: str, column: ChunkedOrArrayAny) -> pa.Table:
    column_names = table.column_names
    if name in column_names:
        return table.set_column(column_names.index(name), name, column)
    return table.append_column(name, column)


def with_arrays(
    table: pa.Table, names_and_columns: Iterable[tuple[str, ChunkedOrArrayAny]], /
) -> pa.Table:
    column_names = table.column_names
    for name, column in names_and_columns:
        if name in column_names:
            table = table.set_column(column_names.index(name), name, column)
        else:
            table = table.append_column(name, column)
    return table
