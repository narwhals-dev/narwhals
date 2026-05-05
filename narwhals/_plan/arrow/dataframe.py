from __future__ import annotations

import operator
from collections.abc import Collection, Iterable
from functools import reduce
from itertools import chain
from typing import TYPE_CHECKING, Any, cast, overload

import pyarrow as pa
import pyarrow.compute as pc

from narwhals._arrow.utils import native_to_narwhals_dtype
from narwhals._plan._namespace import namespace
from narwhals._plan._version import into_version
from narwhals._plan.arrow import acero, compat, functions as fn, io
from narwhals._plan.arrow.common import ArrowFrameSeries as FrameSeries
from narwhals._plan.arrow.expr import ArrowExpr as Expr, ArrowScalar as Scalar
from narwhals._plan.arrow.group_by import (
    ArrowGroupBy as GroupBy,
    named_ir_agg,
    partition_by,
    unique_keep_boolean_length_preserving,
)
from narwhals._plan.arrow.namespace import ArrowNamespace
from narwhals._plan.arrow.pivot import pivot_table
from narwhals._plan.common import temp
from narwhals._plan.compliant.dataframe import EagerDataFrame
from narwhals._plan.exceptions import shape_error
from narwhals._utils import generate_repr, requires, supports_arrow_c_stream

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence
    from io import BytesIO

    import pandas as pd
    import polars as pl
    from typing_extensions import Self, TypeAlias

    from narwhals._plan.arrow.series import ArrowSeries as Series
    from narwhals._plan.arrow.typing import (
        ChunkedArrayAny,
        ChunkedOrArrayAny,
        CompliantSeries,
        IOSource,
        Predicate,
    )
    from narwhals._plan.compliant.dataframe import CompliantDataFrame
    from narwhals._plan.compliant.group_by import GroupByResolver
    from narwhals._plan.compliant.typing import LazyFrameAny
    from narwhals._plan.dataframe import DataFrame as NwDataFrame
    from narwhals._plan.expressions import NamedIR
    from narwhals._plan.options import ExplodeOptions, SortMultipleOptions
    from narwhals._plan.typing import NonCrossJoinStrategy, Seq
    from narwhals._translate import IntoArrowTable
    from narwhals._typing import _LazyAllowedImpl
    from narwhals.dtypes import DType
    from narwhals.typing import (
        AsofJoinStrategy,
        FileSource,
        IntoSchema,
        PivotAgg,
        UniqueKeepStrategy,
    )

Incomplete: TypeAlias = Any


class ArrowDataFrame(
    FrameSeries["pa.Table"], EagerDataFrame["pa.Table", "ChunkedArrayAny"]
):
    def __narwhals_namespace__(self) -> ArrowNamespace:
        return ArrowNamespace()

    def __repr__(self) -> str:
        return generate_repr(f"nw.{type(self).__name__}", self.native.__repr__())

    @classmethod
    def from_native(cls, native: pa.Table, /) -> Self:
        obj = cls.__new__(cls)
        obj._native = native
        return obj

    def _with_native(self, native: pa.Table) -> Self:
        return self.from_native(native)

    @property
    def _group_by(self) -> type[GroupBy]:
        return GroupBy

    @property
    def shape(self) -> tuple[int, int]:
        return self.native.shape

    def lazy(self, backend: _LazyAllowedImpl | None, **kwds: Any) -> LazyFrameAny:
        msg = "ArrowDataFrame.lazy"
        raise NotImplementedError(msg)

    def group_by_resolver(self, resolver: GroupByResolver, /) -> GroupBy:
        return self._group_by.from_resolver(self, resolver)

    @property
    def columns(self) -> list[str]:
        return self.native.column_names

    @property
    def schema(self) -> dict[str, DType]:
        schema = self.native.schema
        return {
            name: native_to_narwhals_dtype(dtype, self.version)
            for name, dtype in zip(schema.names, schema.types)
        }

    def __len__(self) -> int:
        return self.native.num_rows

    @classmethod
    def concat_diagonal(cls, dfs: Iterable[Self]) -> Self:
        return cls.from_native(fn.concat_tables((df.native for df in dfs), "default"))

    @classmethod
    def concat_horizontal(cls, dfs: Iterable[Self]) -> Self:
        return cls.from_native(fn.concat_tables_horizontal(df.native for df in dfs))

    @classmethod
    def concat_vertical(cls, dfs: Iterable[Self]) -> Self:
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
        return cls.from_native(fn.concat_tables(df.native for df in dfs))

    @classmethod
    def concat_series(cls, series: Iterable[CompliantSeries]) -> Self:
        """Used for `ArrowExpr.sort_by`, seems like only pandas needs `stack_horizontal`?"""
        if isinstance(series, Collection):
            arrays, names = [s.native for s in series], [s.name for s in series]
        else:
            arrays, names = [], []
            for s in series:
                arrays.append(s.native)
                names.append(s.name)
        return cls.from_native(fn.concat_horizontal(arrays, names))

    @classmethod
    def from_arrow(cls, data: IntoArrowTable, /) -> Self:
        if isinstance(data, pa.Table):
            native = data
        elif compat.BACKEND_VERSION >= (14,) or isinstance(data, Collection):
            native = pa.table(data)
        elif supports_arrow_c_stream(data):  # pragma: no cover
            msg = f"'pyarrow>=14.0.0' is required for `from_arrow` for object of type {type(data).__name__!r}."
            raise ModuleNotFoundError(msg)
        else:  # pragma: no cover
            msg = f"`from_arrow` is not supported for object of type {type(data).__name__!r}."
            raise TypeError(msg)
        return cls.from_native(native)

    @classmethod
    def from_pandas(cls, frame: pd.DataFrame, /) -> Self:
        return cls.from_native(pa.Table.from_pandas(frame))

    @classmethod
    def from_compliant(
        cls, frame: pl.DataFrame | NwDataFrame[Any, Any] | CompliantDataFrame[Any, Any], /
    ) -> Self:
        return cls.from_native(frame.to_arrow())

    from_polars = from_narwhals = from_compliant

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any], /, *, schema: IntoSchema | None = None
    ) -> Self:
        pa_schema = (
            schema
            if schema is None
            else into_version(cls.version).schema(schema).to_arrow()
        )
        native = pa.Table.from_pydict(data, schema=pa_schema)
        return cls.from_native(native)

    @classmethod
    def read_csv(cls, source: FileSource, /, **kwds: Any) -> Self:
        return cls.from_native(io.read_csv(source, **kwds))

    @classmethod
    def read_parquet(cls, source: IOSource, /, **kwds: Any) -> Self:
        return cls.from_native(io.read_parquet(source, **kwds))

    def _iter_columns(self) -> Iterator[tuple[str, ChunkedArrayAny]]:
        return zip(self.native.column_names, self.native.itercolumns())

    def iter_columns(self) -> Iterator[Series]:
        series_ = namespace(self)._series
        for name, series in self._iter_columns():
            yield series_.from_native(series, name)

    def to_series(self, index: int = 0) -> Series:
        return self.get_column(self.columns[index])

    def to_arrow(self) -> pa.Table:
        return self.native

    def to_pandas(self) -> pd.DataFrame:
        return self.native.to_pandas()

    def to_polars(self) -> pl.DataFrame:
        import polars as pl  # ignore-banned-import
        # NOTE: Recommended in https://github.com/pola-rs/polars/issues/22921#issuecomment-2908506022

        return pl.DataFrame(self.native)

    def _evaluate_irs(
        self, nodes: Iterable[NamedIR], /, *, length: int | None = None
    ) -> Iterator[CompliantSeries]:
        ns = namespace(self)
        expr = ns._expr
        from_named_ir = ns.from_named_ir
        yield from expr.align((from_named_ir(e, self) for e in nodes), default=length)

    def select(self, irs: Seq[NamedIR]) -> Self:
        return self.concat_series(self._evaluate_irs(irs))

    def with_columns(self, irs: Seq[NamedIR]) -> Self:
        return self.concat_series(self._evaluate_irs(irs, length=len(self)))

    def sort(self, by: Sequence[str], options: SortMultipleOptions | None = None) -> Self:
        return self.gather(fn.sort_indices(self.native, *by, options=options))

    def _unique(
        self,
        subset: Sequence[str] | None = None,
        *,
        order_by: Sequence[str] = (),
        keep: UniqueKeepStrategy = "any",
        **_: Any,
    ) -> Self:
        """Drop duplicate rows from this DataFrame.

        Always maintains order, via `with_row_index(_by)`.
        See [`unsort_indices`] for an example.

        [`unsort_indices`]: https://github.com/narwhals-dev/narwhals/blob/9b9122b4ab38a6aebe2f09c29ad0f6191952a7a7/narwhals/_plan/arrow/functions.py#L1666-L1697
        """
        subset = tuple(subset or self.columns)
        into_agg, mask = unique_keep_boolean_length_preserving(keep)
        idx_name = temp.column_name(self.columns)
        df = self.select_names(*set(subset).union(order_by))
        if order_by:
            df = df.with_row_index_by(idx_name, order_by)
        else:
            df = df.with_row_index(idx_name)
        idx_agg = (
            df.group_by_names(subset)
            .agg((named_ir_agg(idx_name, into_agg),))
            .get_column(idx_name)
            .native
        )
        return self._filter(mask(df.get_column(idx_name).native, idx_agg))

    unique = _unique
    unique_by = _unique

    def unpivot(
        self,
        on: Sequence[str] | None,
        index: Sequence[str] | None,
        *,
        variable_name: str = "variable",
        value_name: str = "value",
    ) -> Self:
        n = len(self)
        index = [] if index is None else list(index)
        on_ = (c for c in self.columns if c not in index) if on is None else iter(on)
        index_cols = self.native.select(index)
        column = self.native.column
        tables = (
            index_cols.append_column(variable_name, fn.repeat(name, n)).append_column(
                value_name, column(name)
            )
            for name in on_
        )
        return self._with_native(fn.concat_tables(tables, "permissive"))

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

    @overload
    def write_csv(self, target: None, /) -> str: ...
    @overload
    def write_csv(self, target: str | BytesIO, /) -> None: ...
    def write_csv(self, target: str | BytesIO | None, /) -> str | None:
        import pyarrow.csv as pa_csv

        if target is None:
            csv_buffer = pa.BufferOutputStream()
            pa_csv.write_csv(self.native, csv_buffer)
            return csv_buffer.getvalue().to_pybytes().decode()
        pa_csv.write_csv(self.native, target)
        return None

    def write_parquet(self, target: str | BytesIO, /) -> None:
        import pyarrow.parquet as pp

        pp.write_table(self.native, target)

    def to_struct(self, name: str = "") -> Series:
        native = self.native
        if compat.TO_STRUCT_ARRAY_ACCEPTS_EMPTY:
            struct = native.to_struct_array()
        elif compat.HAS_FROM_TO_STRUCT_ARRAY:
            if len(native):
                struct = native.to_struct_array()
            else:
                struct = fn.chunked_array([], pa.struct(native.schema))
        else:
            struct = fn.struct.into_struct(native.columns, native.column_names)
        return namespace(self)._series.from_native(struct, name)

    def unnest(self, columns: Sequence[str]) -> Self:
        if len(columns) == 1:
            native = self.native
            index = native.column_names.index(columns[0])
            ca_struct = native.column(index)
            arrays: list[ChunkedArrayAny] = ca_struct.flatten()
            names = fn.struct.field_names(ca_struct)
            if len(names) == 1:
                result = native.set_column(index, names[0], arrays[0])
            else:
                result = insert_arrays_at(
                    native.remove_column(index), index, names, arrays
                )
            return self._with_native(result)
        # NOTE: `pa.Table.from_pydict` internally calls `pa.Table.from_arrays`
        to_unnest = frozenset(columns)
        arrays = []
        names = []
        for name, ca in self._iter_columns():
            if name in to_unnest:
                arrays.extend(ca.flatten())
                names.extend(fn.struct.field_names(ca))
            else:
                arrays.append(ca)
                names.append(name)
        return self._with_native(pa.Table.from_arrays(arrays, names))

    def get_column(self, name: str) -> Series:
        chunked = self.native.column(name)
        return namespace(self)._series.from_native(chunked, name)

    def drop(self, columns: Sequence[str]) -> Self:
        return self._with_native(self.native.drop(list(columns)))

    def drop_nulls(self, subset: Sequence[str] | None) -> Self:
        if subset is None:
            native = self.native.drop_null()
        else:
            to_drop = reduce(operator.or_, (pc.field(name).is_null() for name in subset))
            native = self.native.filter(~to_drop)
        return self._with_native(native)

    def explode(self, columns: Sequence[str], options: ExplodeOptions) -> Self:
        builder = fn.ExplodeBuilder.from_options(options)
        if len(columns) == 1:
            return self._with_native(builder.explode_column(self.native, columns[0]))
        return self._with_native(builder.explode_columns(self.native, columns))

    def rename(self, mapping: Mapping[str, str]) -> Self:
        names: dict[str, str] | list[str]
        if compat.TABLE_RENAME_ACCEPTS_DICT:
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

    @requires.backend_version((16,))
    def join_asof(
        self,
        other: Self,
        *,
        left_on: str,
        right_on: str,
        left_by: Sequence[str] = (),
        right_by: Sequence[str] = (),
        strategy: AsofJoinStrategy = "backward",
        suffix: str = "_right",
    ) -> Self:
        return self._with_native(
            acero.join_asof_tables(
                self.native,
                other.native,
                left_on,
                right_on,
                left_by=left_by,
                right_by=right_by,
                strategy=strategy,
                suffix=suffix,
            )
        )

    def _filter(self, predicate: Predicate | acero.Expr) -> Self:
        mask: Incomplete = predicate
        return self._with_native(self.native.filter(mask))

    # NOTE: `Self` cannot be used here while  `ct.Frame` is invariant
    def filter(self, predicate: NamedIR) -> ArrowDataFrame:
        mask: pc.Expression | ChunkedArrayAny
        resolved = predicate.dispatch(namespace(self), self)
        if isinstance(resolved, Expr):
            mask = resolved.broadcast(len(self)).native
        else:
            mask = acero.lit(resolved.native)
        return self._filter(mask)

    def partition_by(self, by: Sequence[str], *, include_key: bool = True) -> list[Self]:
        from_native = self._with_native
        partitions = partition_by(self.native, by, include_key=include_key)
        return [from_native(df) for df in partitions]

    def pivot(
        self,
        on: Sequence[str],
        on_columns: Self,
        *,
        index: Sequence[str],
        values: Sequence[str],
        aggregate_function: PivotAgg | None = None,
        separator: str = "_",
        sort_columns: bool = False,  # polars compat
    ) -> Self:
        result = pivot_table(
            self.native,
            list(on),
            on_columns.native,
            index,
            values,
            aggregate_function,
            separator,
        )
        return self._with_native(result)

    def clone(self) -> Self:
        return self._with_native(self.native)


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


def insert_arrays_at(
    table: pa.Table,
    index: int,
    names: Collection[str],
    columns: Iterable[ChunkedOrArrayAny],
    /,
) -> pa.Table:
    """Add multiple columns to a table, starting at `index`."""
    if index in {0, table.num_columns}:
        if index == 0:
            arrays = (*columns, *table.columns)
            names = (*names, *table.column_names)
        else:
            arrays = (*table.columns, *columns)
            names = (*table.column_names, *names)
        return fn.concat_horizontal(arrays, names)
    for idx, name, column in zip(range(index, index + len(names)), names, columns):
        table = table.add_column(idx, name, column)
    return table


ArrowDataFrame()
