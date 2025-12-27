from __future__ import annotations

import operator
from functools import reduce
from itertools import chain, product
from typing import TYPE_CHECKING, Any, Literal, cast, overload

import pyarrow as pa  # ignore-banned-import
import pyarrow.compute as pc  # ignore-banned-import

from narwhals._arrow.utils import native_to_narwhals_dtype
from narwhals._plan.arrow import (
    acero,
    compat,
    functions as fn,
    group_by,
    options as pa_options,
)
from narwhals._plan.arrow.common import ArrowFrameSeries as FrameSeries
from narwhals._plan.arrow.expr import ArrowExpr as Expr, ArrowScalar as Scalar
from narwhals._plan.arrow.group_by import ArrowGroupBy as GroupBy, partition_by
from narwhals._plan.arrow.series import ArrowSeries as Series
from narwhals._plan.common import temp
from narwhals._plan.compliant.dataframe import EagerDataFrame
from narwhals._plan.compliant.typing import LazyFrameAny, namespace
from narwhals._plan.exceptions import shape_error
from narwhals._plan.expressions import NamedIR, named_ir
from narwhals._utils import Version, generate_repr
from narwhals.schema import Schema

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence
    from io import BytesIO

    import polars as pl
    from typing_extensions import Self, TypeAlias

    from narwhals._plan.arrow.typing import (
        ChunkedArrayAny,
        ChunkedOrArrayAny,
        ChunkedStruct,
        Predicate,
        StructArray,
    )
    from narwhals._plan.compliant.group_by import GroupByResolver
    from narwhals._plan.expressions import ExprIR, NamedIR
    from narwhals._plan.options import ExplodeOptions, SortMultipleOptions
    from narwhals._plan.typing import NonCrossJoinStrategy
    from narwhals._typing import _LazyAllowedImpl
    from narwhals.dtypes import DType
    from narwhals.typing import IntoSchema, PivotAgg, UniqueKeepStrategy

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
        into_column_agg, mask = fn.unique_keep_boolean_length_preserving(keep)
        idx_name = temp.column_name(self.columns)
        df = self.select_names(*set(subset).union(order_by))
        if order_by:
            df = df.with_row_index_by(idx_name, order_by)
        else:
            df = df.with_row_index(idx_name)
        idx_agg = (
            df.group_by_names(subset)
            .agg((named_ir(idx_name, into_column_agg(idx_name)),))
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

    def _filter(self, predicate: Predicate | acero.Expr) -> Self:
        mask: Incomplete = predicate
        return self._with_native(self.native.filter(mask))

    def filter(self, predicate: NamedIR) -> Self:
        mask: pc.Expression | ChunkedArrayAny
        resolved = Expr.from_named_ir(predicate, self)
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
        separator: str = "_",
    ) -> Self:
        native = self.native
        on_columns_ = on_columns.native
        if len(on) == 1:
            pivot = acero.pivot_table(native, on[0], on_columns_.column(0), index, values)
        else:
            # implode every `values` column, within the pivot groups
            specs = (group_by.AggSpec(value, "hash_list") for value in values)
            pre_agg = acero.group_by_table(native, [*index, *on], specs)
            # NOTE: The actual `pivot(on)` we pass to `pyarrow` is an index into the groups produced by `on: list[str]`
            on_columns_encoded = fn.int_range(len(pre_agg))
            temp_name = temp.column_name(native.column_names)
            pre_agg_w_idx = pre_agg.add_column(0, temp_name, on_columns_encoded)

            post_explode = fn.ExplodeBuilder().explode_columns(
                pre_agg_w_idx.select([temp_name, *index, *values]), values
            )

            # this also does a similar thing to `pivot_agg`, for `on_columns`
            pivot = acero.pivot_table(
                post_explode, temp_name, on_columns_encoded, index, values
            )
        result = _temp_post_pivot_table(pivot, on_columns_, index, values, separator)
        return self._with_native(result)

    # TODO @dangotbanned: Align each of the impls more, then de-duplicate
    def pivot_agg(
        self,
        on: Sequence[str],
        on_columns: Self,
        *,
        index: Sequence[str],
        values: Sequence[str],
        aggregate_function: PivotAgg,
        separator: str = "_",
    ) -> Self:
        native = self.native
        tp_agg = group_by.SUPPORTED_PIVOT_AGG[aggregate_function]
        agg_func = group_by.SUPPORTED_AGG[tp_agg]
        option = pa_options.AGG.get(tp_agg)
        specs = (group_by.AggSpec(value, agg_func, option) for value in values)

        if len(on) == 1:
            pre_agg = acero.group_by_table(native, [*index, *on], specs)
            return self._with_native(pre_agg).pivot(
                on, on_columns, index=index, values=values, separator=separator
            )
        temp_name = temp.column_name(native.column_names)
        on_columns_w_idx = on_columns.with_row_index(temp_name)
        on_columns_encoded = on_columns_w_idx.get_column(temp_name).native
        single_on = self.join_inner(on_columns_w_idx, list(on)).drop(on).native
        pre_agg = acero.group_by_table(single_on, [*index, temp_name], specs)

        # this part is the tricky one, since the pivot and the renaming use different reprs for `on_columns`
        pivot = acero.pivot_table(pre_agg, temp_name, on_columns_encoded, index, values)
        result = _temp_post_pivot_table(
            pivot, on_columns.native, index, values, separator
        )
        return self._with_native(result)


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


def struct_to_arrays(native: ChunkedStruct | StructArray) -> Sequence[ChunkedOrArrayAny]:
    """Unnest the fields of a struct into one array per-struct-field.

    Cheaper than `unnest`-ing into a `Table`, and very helpful if the names are going to be replaced.
    """
    return cast("ChunkedStruct | pa.StructArray", native).flatten()


@overload
def structs_to_arrays(
    *structs: ChunkedStruct | StructArray,
) -> Iterator[Sequence[ChunkedOrArrayAny]]: ...
@overload
def structs_to_arrays(
    *structs: ChunkedStruct | StructArray, flatten: Literal[True]
) -> Iterator[ChunkedOrArrayAny]: ...
def structs_to_arrays(
    *structs: ChunkedStruct | StructArray, flatten: bool = False
) -> Iterator[Sequence[ChunkedOrArrayAny] | ChunkedOrArrayAny]:
    """Unnest the fields of every struct into one array per-struct-field.

    By default, yields the arrays of each struct *as a group*, configurable via `flatten`.

    Arguments:
        *structs: One or more Struct-typed arrow arrays.
        flatten: Yield each array from each struct *without grouping*.
    """
    if flatten:
        for struct in structs:
            yield from struct_to_arrays(struct)
    else:
        for struct in structs:
            yield struct_to_arrays(struct)


def _on_columns_names(
    on_columns: pa.Table, values: Sequence[str], *, separator: str = "_"
) -> Iterable[str]:
    """Alignment to polars pivot column naming conventions.

    If we started with:

        {'on_lower': ['b', 'a', 'b', 'a'], 'on_upper': ['X', 'X', 'Y', 'Y']}

    Then this operation will return:

        ['{"b","X"}', '{"a","X"}', '{"b","Y"}', '{"a","Y"}']
    """
    result: Iterable[Any]
    if on_columns.num_columns == 1:
        on_column = on_columns.column(0)
        if len(values) == 1:
            result = on_column.to_pylist()
        else:
            t_left = fn.to_table(fn.array(values))
            # NOTE: still don't know why pyarrow outputs the cross join in reverse
            t_right = fn.to_table(on_column[::-1])
            cross_joined = acero.join_cross_tables(t_left, t_right)
            result = fn.concat_str(*cross_joined.columns, separator=separator).to_pylist()
    else:
        result = fn.concat_str(
            '{"', fn.concat_str(*on_columns.columns, separator='","'), '"}'
        ).to_pylist()
        if len(values) != 1:
            return (
                f"{value}{separator}{name}" for value, name in product(values, result)
            )
    return cast("list[str]", result)


# TODO @dangotbanned: Remember, temporary!
def _temp_post_pivot_table(
    pivot: pa.Table,
    on_columns: pa.Table,
    index: Sequence[str],
    values: Sequence[str],
    separator: str = "_",
) -> pa.Table:
    """Everything here should be moved to `acero.pivot_table`."""
    pivot_columns = pivot.columns
    n_index = len(index)
    unnested = structs_to_arrays(*pivot_columns[n_index:], flatten=True)
    names_final = (*index, *_on_columns_names(on_columns, values, separator=separator))
    return fn.concat_horizontal((*pivot_columns[:n_index], *unnested), names_final)
