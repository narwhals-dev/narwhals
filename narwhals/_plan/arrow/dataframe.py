from __future__ import annotations

import operator
from collections import deque
from functools import reduce
from itertools import chain, product
from typing import TYPE_CHECKING, Any, Literal, cast, overload

import pyarrow as pa  # ignore-banned-import
import pyarrow.compute as pc  # ignore-banned-import

from narwhals._arrow.utils import native_to_narwhals_dtype
from narwhals._plan.arrow import acero, compat, functions as fn, options as pa_options
from narwhals._plan.arrow.common import ArrowFrameSeries as FrameSeries
from narwhals._plan.arrow.expr import ArrowExpr as Expr, ArrowScalar as Scalar
from narwhals._plan.arrow.group_by import ArrowGroupBy as GroupBy, partition_by
from narwhals._plan.arrow.series import ArrowSeries as Series
from narwhals._plan.common import temp, todo
from narwhals._plan.compliant.dataframe import EagerDataFrame
from narwhals._plan.compliant.typing import LazyFrameAny, namespace
from narwhals._plan.exceptions import shape_error
from narwhals._plan.expressions import NamedIR, named_ir
from narwhals._utils import Version, generate_repr
from narwhals.exceptions import InvalidOperationError
from narwhals.schema import Schema

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Collection,
        Iterable,
        Iterator,
        Mapping,
        Sequence,
    )
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
    from narwhals.typing import IntoSchema, UniqueKeepStrategy

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
        on_columns: Sequence[str] | Self,
        *,
        index: Sequence[str],
        values: Sequence[str],
        separator: str = "_",
    ) -> Self:
        if isinstance(on_columns, ArrowDataFrame):
            return self._with_native(
                pivot_on_multiple(
                    self.native,
                    on,
                    on_columns.native,
                    index=index,
                    values=values,
                    separator=separator,
                )
            )
        if len(on) == 1:
            return self._with_native(
                pivot(
                    self.native,
                    on[0],
                    on_columns,
                    index=index,
                    values=values,
                    separator=separator,
                )
            )
        # TODO @dangotbanned: Handle this more gracefully at the `narwhals`-level
        msg = (
            f"Invalid argument combination:\n    `pivot({on=}, {on_columns=})`\n\n"
            "`on_columns` cannot currently be used with multiple `on` names."
        )
        raise InvalidOperationError(msg)

    pivot_agg = todo()


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


def pivot_on_multiple(
    native: pa.Table,
    on: Sequence[str],
    on_columns: pa.Table,
    *,
    index: Sequence[str],
    values: Sequence[str],
    separator: str = "_",  # separator would be used if multiple `values` were handled (todo)  # noqa: ARG001
) -> pa.Table:
    on_columns_names = _on_columns_multiple_names(on_columns, values)
    # bad name
    index = list(index)
    mid_pivot = (
        native.group_by([*index, *on])
        .aggregate([(value, "hash_list", None) for value in values])
        .rename_columns([*index, *on, *values])
    )
    column_index = fn.int_range(len(mid_pivot) * len(values))
    temp_name = temp.column_name(native.column_names)
    mid_pivot_w_idx = mid_pivot.add_column(0, temp_name, column_index)

    agg_name: Any = "hash_pivot_wider"
    options = pa_options.pivot_wider(column_index.cast(pa.string()).to_pylist())
    specs = [([temp_name, value], agg_name, options) for value in values]
    # NOTE: not sure if needed, but helping explore what should happen
    post_explode = fn.ExplodeBuilder().explode_columns(
        mid_pivot_w_idx.select([temp_name, *index, *values]), values
    )

    result = post_explode.group_by(index).aggregate(specs)
    structs = result.columns[len(index) :]

    unnested = chain.from_iterable(structs_to_arrays(*structs))
    arrays_final = (*result.select(index).columns, *unnested)
    names_final = (*index, *on_columns_names)
    return fn.concat_horizontal(arrays_final, names_final)


# TODO @dangotbanned: Is pre/post-aggregating even possible?
def pivot(
    native: pa.Table,
    on: str,
    on_columns: Sequence[str],
    *,
    index: Sequence[str],
    values: Sequence[str],
    separator: str = "_",
) -> pa.Table:
    # pyarrow-stubs doesn't include pivot yet
    agg_name: Any = "hash_pivot_wider"
    options = pa_options.pivot_wider(on_columns)
    specs = [([on, value], agg_name, options) for value in values]
    result = native.group_by(list(index)).aggregate(specs)

    split_index = result.schema.get_field_index(index[-1]) + 1
    result_columns = result.columns
    unnesting, names = _iter_unnest_with_names(result_columns[split_index:])
    arrays = (*result_columns[:split_index], *unnesting)
    if (n := len(values)) > 1:
        # NOTE: May need to approach differently for `on: list[str]`
        # `names` would be cycled `n` times, so we drop everything outside the first cycle
        it_renames = (
            f"{value}{separator}{name}"
            for value, name in product(values, deque(names, maxlen=n))
        )
        names = (*index, *it_renames)
    else:
        names = (*index, *names)
    return fn.concat_horizontal(arrays, names)


# TODO @dangotbanned: Pass `values` for context on how to store `names`
# - `deque` only really makes sense in the `values: str` case
# - if you flatten out each `column_names`, the iterator can splat into `dict.fromkeys`,
#   to handle deduplicating in order
def _iter_unnest_with_names(
    arrays: Iterable[ChunkedStruct],
) -> tuple[Iterator[ChunkedArrayAny], Collection[str]]:
    """Probably the least reusable function ever written.

    The idea is the chunked arrays need to be unpacked in `pivot`,
    so here they are flattened as references to iterators.
    """
    names = deque[str]()
    columns: deque[Iterator[ChunkedArrayAny]] = deque()
    for arr in arrays:
        table = _unnest(arr)
        names.extend(table.column_names)
        columns.append(table.itercolumns())
    return chain.from_iterable(columns), names


def struct_to_arrays(native: ChunkedStruct | StructArray) -> Sequence[ChunkedOrArrayAny]:
    """Unnest the fields of a struct into one array per-struct-field.

    Cheaper than `unnest`-ing into a `Table`, and very helpful if the names are going to be replaced.
    """
    return cast("ChunkedStruct | pa.StructArray", native).flatten()


def structs_to_arrays(
    *structs: ChunkedStruct | pa.StructArray,
) -> Iterator[Sequence[ChunkedOrArrayAny]]:
    """Unnest the fields of every struct into one array per-struct-field.

    Probably want to choose between returning `[...]` or `[[...],[...]]`
    """
    for struct in structs:
        yield struct_to_arrays(struct)


def _unnest(ca: ChunkedStruct, /) -> pa.Table:
    # NOTE: This is the most backwards-compatible version of `Series.struct.unnest`
    batch = cast("Callable[[Any], pa.RecordBatch]", pa.RecordBatch.from_struct_array)
    return pa.Table.from_batches((batch(c) for c in ca.chunks), fn.struct_schema(ca))


def _on_columns_multiple_names(on_columns: pa.Table, values: Sequence[str]) -> list[str]:
    """Alignment to polars naming style when `pivot(on: list[str])`.

    If we started with:

        {'on_lower': ['b', 'a', 'b', 'a'], 'on_upper': ['X', 'X', 'Y', 'Y']}

    Then this operation will return:

        ['{"b","X"}', '{"a","X"}', '{"b","Y"}', '{"a","Y"}']
    """
    if on_columns.num_columns < 2:
        msg = "This operation is not required for `pivot(on: str)`"
        raise ValueError(msg)
    if len(values) != 1:
        # NOTE: Total new columns produced: `len(on_columns) * len(values)`
        #                                    ^ unique rows
        msg = (
            "`TODO: pivot(on: list[str], values: list[str])\n"
            '`{"b","X"}` -> `foo_{"b","X"}`, `bar_{"b","X"}`'
        )
        raise NotImplementedError(msg)
    result = fn.concat_str(
        '{"', fn.concat_str(*on_columns.columns, separator='","'), '"}'
    ).to_pylist()
    return cast("list[str]", result)


def _various_notes() -> None:
    """Pivot values according to a pivot key column.

    Output is a struct with as many fields as `key_names`.
    All output struct fields have the same type as `pivot_values`.

    Each pivot key decides in which output field the corresponding pivot value
    is emitted. If a pivot key doesn't appear, null is emitted.
    If more than one non-null value is encountered for a given pivot key,
    Invalid is raised.

    Important:
        Docs here are for the (scalar) [`"pivot_wider"`] function.
        Apparently need to use [`"hash_pivot_wider"`], but doesn't have the same level of docs

    Arguments:
        pivot_keys: Array-like argument to compute function.
        pivot_values: Array-like argument to compute function.
        key_names: The pivot key names expected in the `pivot_keys` column.
            For each entry in `key_names`, a column with the same name is emitted in the struct output.
        unexpected_key_behavior: The behavior when pivot keys not in `key_names` are encountered.
            If “ignore”, unexpected keys are silently ignored.
            If “raise”, unexpected keys raise a KeyError.

    Note:
        Extra stuff for `"hash_pivot_wider"`
        > (7) The first input contains the pivot key, while the second input contains the values to be pivoted.
        > The output is a Struct with one field for each key in `PivotOptions::key_names`.

    Note:
        [discussion_r1974825436]
        > So the pandas's `pivot` is more like our `hash` version of `pivot_wider`?
        > Yes

    [`"pivot_wider"`]: https://arrow.apache.org/docs/dev/python/generated/pyarrow.compute.pivot_wider.html
    [discussion_r1974825436]: https://github.com/apache/arrow/pull/45562#discussion_r1974825436
    [`"hash_pivot_wider"`]: https://arrow.apache.org/docs/dev/cpp/compute.html#grouped-aggregations-group-by
    """
    raise NotImplementedError
