from __future__ import annotations

from itertools import chain
from typing import TYPE_CHECKING, Any, Literal, cast, overload

import pyarrow as pa  # ignore-banned-import
import pyarrow.compute as pc  # ignore-banned-import

from narwhals._plan import expressions as ir
from narwhals._plan._dispatch import get_dispatch_name
from narwhals._plan._guards import is_agg_expr, is_function_expr
from narwhals._plan.arrow import acero, functions as fn, options
from narwhals._plan.common import temp
from narwhals._plan.compliant.group_by import EagerDataFrameGroupBy
from narwhals._plan.expressions import aggregation as agg
from narwhals._utils import Implementation, qualified_type_name
from narwhals.exceptions import InvalidOperationError

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence

    from typing_extensions import Self, TypeAlias

    from narwhals._plan.arrow.dataframe import ArrowDataFrame as Frame
    from narwhals._plan.arrow.typing import (
        ArrayAny,
        ChunkedArray,
        ChunkedArrayAny,
        ChunkedList,
        ChunkedOrScalarAny,
        Indices,
        ListScalar,
        ScalarAny,
    )
    from narwhals._plan.expressions import NamedIR
    from narwhals._plan.typing import Seq

Incomplete: TypeAlias = Any

SUPPORTED_AGG: Mapping[type[agg.AggExpr], acero.Aggregation] = {
    agg.Sum: "hash_sum",
    agg.Mean: "hash_mean",
    agg.Median: "hash_approximate_median",
    agg.Max: "hash_max",
    agg.Min: "hash_min",
    agg.Std: "hash_stddev",
    agg.Var: "hash_variance",
    agg.Count: "hash_count",
    agg.Len: "hash_count",
    agg.NUnique: "hash_count_distinct",
    agg.First: "hash_first",
    agg.Last: "hash_last",
    fn.MinMax: "hash_min_max",
}
SUPPORTED_LIST_AGG: Mapping[type[ir.lists.Aggregation], type[agg.AggExpr]] = {
    ir.lists.Mean: agg.Mean,
    ir.lists.Median: agg.Median,
    ir.lists.Max: agg.Max,
    ir.lists.Min: agg.Min,
    ir.lists.Sum: agg.Sum,
    ir.lists.First: agg.First,
    ir.lists.Last: agg.Last,
    ir.lists.NUnique: agg.NUnique,
}
SUPPORTED_IR: Mapping[type[ir.ExprIR], acero.Aggregation] = {
    ir.Len: "hash_count_all",
    ir.Column: "hash_list",
}

_version_dependent: dict[Any, acero.Aggregation] = {}
if fn.HAS_KURTOSIS_SKEW:
    _version_dependent.update(
        {ir.functions.Kurtosis: "hash_kurtosis", ir.functions.Skew: "hash_skew"}
    )

SUPPORTED_FUNCTION: Mapping[type[ir.Function], acero.Aggregation] = {
    ir.boolean.All: "hash_all",
    ir.boolean.Any: "hash_any",
    ir.functions.Unique: "hash_distinct",
    ir.functions.NullCount: "hash_count",
    **_version_dependent,
}

del _version_dependent


SUPPORTED_LIST_FUNCTION: Mapping[type[ir.lists.Aggregation], type[ir.Function]] = {
    ir.lists.Any: ir.boolean.Any,
    ir.lists.All: ir.boolean.All,
}

SCALAR_OUTPUT_TYPE: Mapping[acero.Aggregation, pa.DataType] = {
    "all": fn.BOOL,
    "any": fn.BOOL,
    "approximate_median": fn.F64,
    "count": fn.I64,
    "count_all": fn.I64,
    "count_distinct": fn.I64,
    "kurtosis": fn.F64,
    "mean": fn.F64,
    "skew": fn.F64,
    "stddev": fn.F64,
    "variance": fn.F64,
}
"""Scalar aggregates that have an output type **not** dependent on input types*.

For use in list aggregates, where the input was null.

*Except `"mean"` will preserve `Decimal`, if that's where we started.
"""


class AggSpec:
    __slots__ = ("_function", "_name", "_option", "_target")

    def __init__(
        self,
        target: acero.Target,
        agg: acero.Aggregation,
        option: acero.Opts = None,
        name: acero.OutputName = "",
    ) -> None:
        self._target = target
        self._function: acero.Aggregation = agg
        self._option: acero.Opts = option
        self._name: acero.OutputName = name or str(target)

    @property
    def use_threads(self) -> bool:
        """See https://github.com/apache/arrow/issues/36709."""
        return acero.can_thread(self._function)

    def __iter__(self) -> Iterator[acero.Target | acero.Aggregation | acero.Opts]:
        """Let's us duck-type as a 4-tuple."""
        yield from (self._target, self._function, self._option, self._name)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._target!r}, {self._function!r}, {self._option!r}, {self._name!r})"

    @classmethod
    def from_named_ir(cls, named_ir: NamedIR) -> Self:
        return cls.from_expr_ir(named_ir.expr, named_ir.name)

    @classmethod
    def from_agg_expr(cls, expr: agg.AggExpr, name: acero.OutputName) -> Self:
        tp = type(expr)
        if not (agg_name := SUPPORTED_AGG.get(tp)):
            raise group_by_error(name, expr)
        if not isinstance(expr.expr, ir.Column):
            raise group_by_error(name, expr, "too complex")
        option = (
            options.variance(expr.ddof)
            if isinstance(expr, (agg.Std, agg.Var))
            else options.AGG.get(tp)
        )
        return cls(expr.expr.name, agg_name, option, name)

    @classmethod
    def from_function_expr(cls, expr: ir.FunctionExpr, name: acero.OutputName) -> Self:
        tp = type(expr.function)
        if not (fn_name := SUPPORTED_FUNCTION.get(tp)):
            raise group_by_error(name, expr)
        args = expr.input
        if not (len(args) == 1 and isinstance(args[0], ir.Column)):
            raise group_by_error(name, expr, "too complex")
        return cls(args[0].name, fn_name, options.FUNCTION.get(tp), name)

    @classmethod
    def from_expr_ir(cls, expr: ir.ExprIR, name: acero.OutputName) -> Self:
        if is_agg_expr(expr):
            return cls.from_agg_expr(expr, name)
        if is_function_expr(expr):
            return cls.from_function_expr(expr, name)
        if not isinstance(expr, (ir.Len, ir.Column)):
            raise group_by_error(name, expr)
        fn_name = SUPPORTED_IR[type(expr)]
        return cls(expr.name if isinstance(expr, ir.Column) else (), fn_name, name=name)

    # NOTE: Fast-paths for single column rewrites
    @classmethod
    def _from_function(cls, tp: type[ir.Function], name: str) -> Self:
        return cls(name, SUPPORTED_FUNCTION[tp], options.FUNCTION.get(tp), name)

    @classmethod
    def _from_list_agg(cls, list_agg: ir.lists.Aggregation, /, name: str) -> Self:
        tp = type(list_agg)
        if tp_agg := SUPPORTED_LIST_AGG.get(tp):
            if tp_agg in {agg.Std, agg.Var}:
                msg = (
                    f"TODO: {qualified_type_name(list_agg)!r} needs access to `ddof`.\n"
                    "Add some sugar around mapping `ListFunction.<parameter>` -> `AggExpr.<parameter>`\n"
                    "or  using `Immutable.__immutable_keys__`"
                )
                raise NotImplementedError(msg)
            fn_name = SUPPORTED_AGG[tp_agg]
        elif tp_func := SUPPORTED_LIST_FUNCTION.get(tp):
            fn_name = SUPPORTED_FUNCTION[tp_func]
        else:
            raise NotImplementedError(tp)
        return cls(name, fn_name, options.LIST_AGG.get(tp), name)

    @classmethod
    def any(cls, name: str) -> Self:
        return cls._from_function(ir.boolean.Any, name)

    @classmethod
    def unique(cls, name: str) -> Self:
        return cls._from_function(ir.functions.Unique, name)

    @classmethod
    def implode(cls, name: str) -> Self:
        # TODO @dangotbanned: Replace with `agg.Implode` (via `_from_agg`) once both have been dded
        # https://github.com/pola-rs/polars/blob/1684cc09dfaa46656dfecc45ab866d01aa69bc78/crates/polars-plan/src/dsl/expr/mod.rs#L44
        return cls(name, SUPPORTED_IR[ir.Column], None, name)

    @overload
    def agg_list(self, native: ChunkedList) -> ChunkedArrayAny: ...
    @overload
    def agg_list(self, native: ListScalar) -> ScalarAny: ...
    def agg_list(self, native: ChunkedList | ListScalar) -> ChunkedOrScalarAny:
        """Execute this aggregation over the values in *each* list, reducing *each* to a single value."""
        result: ChunkedOrScalarAny
        if isinstance(native, pa.Scalar):
            scalar = cast("pa.ListScalar[Any]", native)
            func = HASH_TO_SCALAR_NAME[self._function]
            if not scalar.is_valid:
                return fn.lit(None, SCALAR_OUTPUT_TYPE.get(func, scalar.type.value_type))
            result = pc.call_function(func, [scalar.values], self._option)
            return result
        result = self.over_index(fn.ExplodeBuilder().explode_with_indices(native), "idx")
        result = fn.when_then(native.is_valid(), result)
        if self._is_n_unique():
            # NOTE: Exploding `[]` becomes `[None]` - so we need to adjust the unique count *iff* we were unlucky
            len_not_eq_0 = fn.not_eq(fn.list_len(native), fn.lit(0))
            if not fn.all_(len_not_eq_0, ignore_nulls=False).as_py():
                result = fn.when_then(fn.not_(len_not_eq_0), fn.lit(0), result)
        return result

    def over(self, native: pa.Table, keys: Iterable[acero.Field]) -> pa.Table:
        """Sugar for `native.group_by(keys).aggregate([self])`.

        Returns a table with columns named: `[*keys, self.name]`
        """
        return acero.group_by_table(native, keys, [self])

    def over_index(self, native: pa.Table, index_column: str) -> ChunkedArrayAny:
        """Execute this aggregation over `index_column`.

        Returns a single, (unnamed) array, representing the aggregation results.
        """
        return acero.group_by_table(native, [index_column], [self]).column(self._name)

    def _is_n_unique(self) -> bool:
        return self._function == SUPPORTED_AGG[agg.NUnique]


def group_by_error(
    column_name: str, expr: ir.ExprIR, reason: Literal["too complex"] | None = None
) -> InvalidOperationError:
    backend = Implementation.PYARROW
    if reason == "too complex":
        msg = "Non-trivial complex aggregation found, which"
    else:
        msg = f"`{get_dispatch_name(expr)}()`"
    msg = f"{msg} is not supported in a `group_by` context for {backend!r}:\n{column_name}={expr!r}"
    return InvalidOperationError(msg)


class ArrowGroupBy(EagerDataFrameGroupBy["Frame"]):
    _df: Frame
    _keys: Seq[NamedIR]
    _key_names: Seq[str]
    _key_names_original: Seq[str]

    def __iter__(self) -> Iterator[tuple[Any, Frame]]:
        by = self.key_names
        from_native = self.compliant._with_native
        for partition in partition_by(self.compliant.native, by):
            t = from_native(partition)
            yield (
                t.select_names(*by).row(0),
                t.select_names(*self._column_names_original),
            )

    def agg(self, irs: Seq[NamedIR]) -> Frame:
        compliant = self.compliant
        native = compliant.native
        key_names = self.key_names
        specs = (AggSpec.from_named_ir(e) for e in irs)
        result = compliant._with_native(acero.group_by_table(native, key_names, specs))
        if original := self._key_names_original:
            return result.rename(dict(zip(key_names, original)))
        return result

    def agg_over(self, irs: Seq[NamedIR], sort_indices: Indices | None = None) -> Frame:
        key_names = list(self.key_names)
        compliant = self.compliant
        native = compliant.native
        column_names = native.column_names
        agg_names = (e.name for e in irs)
        from_native = compliant._with_native

        # Handle null values in partitions, trying to avoid any work if possible
        if len(key_names) == 1:
            by = native.column(key_names[0])
            if by.null_count:
                temp_name = temp.column_name({*column_names, *agg_names})
                key_names = [temp_name]
                native = native.append_column(temp_name, dictionary_encode(by))
                compliant = from_native(native)
        else:
            partitions = native.select(key_names)
            it_temp_names = temp.column_names(chain(column_names, agg_names))
            by_names: list[str] = []
            for orig_name, by in zip(key_names, partitions.columns):
                if by.null_count:
                    by_name = next(it_temp_names)
                    native = native.append_column(by_name, dictionary_encode(by))
                else:
                    by_name = orig_name
                by_names.append(by_name)
            if by_names != key_names:
                key_names = by_names
                compliant = from_native(native)

        # If `order_by` was used, we can now apply the new order to the aggregation only
        ordered = native if sort_indices is None else compliant._gather(sort_indices)
        specs = (AggSpec.from_named_ir(e) for e in irs)
        windowed = from_native(acero.group_by_table(ordered, key_names, specs))
        return (
            compliant.select_names(*key_names)
            .join_inner(windowed, key_names)
            .drop(key_names)
        )


@overload
def dictionary_encode(native: ChunkedArrayAny, /) -> pa.Int32Array: ...
@overload
def dictionary_encode(
    native: ChunkedArrayAny, /, *, include_values: Literal[True]
) -> tuple[ArrayAny, pa.Int32Array]: ...
def dictionary_encode(
    native: ChunkedArrayAny, /, *, include_values: bool = False
) -> tuple[ArrayAny, pa.Int32Array] | pa.Int32Array:
    """Extra typing for `pc.dictionary_encode`."""
    da: Incomplete = native.dictionary_encode("encode").combine_chunks()
    indices: pa.Int32Array = da.indices
    if not include_values:
        return indices
    values: ArrayAny = da.dictionary
    return values, indices


def _composite_key(native: pa.Table, *, separator: str = "") -> ChunkedArray:
    """Horizontally join columns to *seed* a unique key per row combination."""
    dtype = fn.string_type(native.schema.types)
    it = fn.cast_table(native, dtype).itercolumns()
    concat: Incomplete = pc.binary_join_element_wise
    join = options.join_replace_nulls()
    return concat(*it, fn.lit(separator, dtype), options=join)  # type: ignore[no-any-return]


def partition_by(
    native: pa.Table, by: Sequence[str], *, include_key: bool = True
) -> Iterator[pa.Table]:
    if len(by) == 1:
        yield from _partition_by_one(native, by[0], include_key=include_key)
    else:
        yield from _partition_by_many(native, by, include_key=include_key)


def _partition_by_one(
    native: pa.Table, by: str, *, include_key: bool = True
) -> Iterator[pa.Table]:
    """Optimized path for single-column partition."""
    values, indices = dictionary_encode(native.column(by), include_values=True)
    if not include_key:
        native = native.remove_column(native.schema.get_field_index(by))
    for idx in range(len(values)):
        # NOTE: Acero filter doesn't support `null_selection_behavior="emit_null"`
        # Is there any reasonable way to do this in Acero?
        yield native.filter(pc.equal(pa.scalar(idx), indices))


def _partition_by_many(
    native: pa.Table, by: Sequence[str], *, include_key: bool = True
) -> Iterator[pa.Table]:
    original_names = native.column_names
    temp_name = temp.column_name(original_names)
    key = acero.col(temp_name)
    composite_values = _composite_key(acero.select_names_table(native, by))
    # Need to iterate over the whole thing, so py_list first should be faster
    unique_py = composite_values.unique().to_pylist()
    re_keyed = native.add_column(0, temp_name, composite_values)
    source = acero.table_source(re_keyed)
    if include_key:
        keep = original_names
    else:
        ignore = {*by, temp_name}
        keep = [name for name in original_names if name not in ignore]
    select = acero.select_names(keep)
    for v in unique_py:
        # NOTE: May want to split the `Declaration` production iterator into it's own function
        # E.g, to push down column selection to *before* collection
        # Not needed for this task though
        yield acero.collect(source, acero.filter(key == v), select)


def _generate_hash_to_scalar_name() -> Mapping[acero.Aggregation, acero.Aggregation]:
    nw_to_hash = SUPPORTED_AGG, SUPPORTED_IR, SUPPORTED_FUNCTION
    only_hash = {"hash_distinct", "hash_list", "hash_one"}
    targets = set[str](chain.from_iterable(m.values() for m in nw_to_hash)) - only_hash
    hash_to_scalar = {hash_name: hash_name.removeprefix("hash_") for hash_name in targets}
    # NOTE: Support both of these when using `AggSpec` directly for scalar aggregates
    # `(..., "hash_mean", ..., ...)`
    # `(..., "mean", ..., ...)`
    scalar_names = hash_to_scalar.values()
    scalar_to_scalar = zip(scalar_names, scalar_names)
    hash_to_scalar.update(dict(scalar_to_scalar))
    return cast("Mapping[acero.Aggregation, acero.Aggregation]", hash_to_scalar)


# TODO @dangotbanned: Replace this with a lazier version
# Don't really want this running at import-time, but using `ModuleType.__getattr__` means
# defining it somewhere else
HASH_TO_SCALAR_NAME: Mapping[acero.Aggregation, acero.Aggregation] = (
    _generate_hash_to_scalar_name()
)
"""Mapping between [Hash aggregate] and [Scalar aggregate] names.

Dynamically built for use in `ListScalar` aggregations, accounting for version availability.

[Hash aggregate]: https://arrow.apache.org/docs/dev/cpp/compute.html#grouped-aggregations-group-by
[Scalar aggregate]: https://arrow.apache.org/docs/dev/cpp/compute.html#aggregations
"""
