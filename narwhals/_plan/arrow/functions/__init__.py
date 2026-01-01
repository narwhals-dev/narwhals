"""Native functions, aliased and/or with behavior aligned to `polars`.

- [x] _categorical -> `cat`
- [x] _construction
- [x] _dtypes
- [x] _ranges
- [x] _repeat
- [ ] _strings
  - [x] -> `str_` (until `functions.__init__` is cleaner)
  - [ ] -> `str`
- [ ] _lists -> `list`
- [x] _struct -> `struct`
- [x] _bin_op
- [x] _boolean
- [ ] _aggregation
- [ ] ...
"""

from __future__ import annotations

import math
import typing as t
from itertools import chain
from typing import TYPE_CHECKING, Any, Final, Literal, overload

import pyarrow as pa  # ignore-banned-import
import pyarrow.compute as pc  # ignore-banned-import

from narwhals._plan._guards import is_non_nested_literal
from narwhals._plan.arrow import compat, options as pa_options
from narwhals._plan.arrow.functions import (  # noqa: F401
    _categorical as cat,
    _strings as str_,
    _struct as struct,
)
from narwhals._plan.arrow.functions._bin_op import (
    add as add,
    and_ as and_,
    binary as binary,
    eq as eq,
    floordiv as floordiv,
    gt as gt,
    gt_eq as gt_eq,
    lt as lt,
    lt_eq as lt_eq,
    modulus as modulus,
    multiply as multiply,
    not_eq as not_eq,
    or_ as or_,
    power as power,
    sub as sub,
    truediv as truediv,
    xor as xor,
)
from narwhals._plan.arrow.functions._boolean import (
    BOOLEAN_LENGTH_PRESERVING as BOOLEAN_LENGTH_PRESERVING,
    all_ as all_,  # TODO @dangotbanned: Import as `all` when namespace is cleaner
    any_ as any_,  # TODO @dangotbanned: Import as `any` when namespace is cleaner
    eq_missing as eq_missing,
    is_between as is_between,
    is_finite as is_finite,
    is_in as is_in,
    is_nan as is_nan,
    is_not_nan as is_not_nan,
    is_not_null as is_not_null,
    is_null as is_null,
    is_only_nulls as is_only_nulls,
    not_ as not_,
    unique_keep_boolean_length_preserving as unique_keep_boolean_length_preserving,
)
from narwhals._plan.arrow.functions._common import reverse as reverse
from narwhals._plan.arrow.functions._construction import (
    array as array,
    chunked_array as chunked_array,
    concat_horizontal as concat_horizontal,
    concat_tables as concat_tables,
    concat_vertical as concat_vertical,
    lit as lit,
    to_table as to_table,
)
from narwhals._plan.arrow.functions._cumulative import (
    cum_count as cum_count,
    cum_max as cum_max,
    cum_min as cum_min,
    cum_prod as cum_prod,
    cum_sum as cum_sum,
    cumulative as cumulative,
)
from narwhals._plan.arrow.functions._dtypes import (
    BOOL as BOOL,
    DATE32 as DATE32,
    F64 as F64,
    I32 as I32,
    I64 as I64,
    UI32 as UI32,
    cast as cast,
    cast_table as cast_table,
    dtype_native as dtype_native,
    string_type as string_type,
)
from narwhals._plan.arrow.functions._ranges import (
    date_range as date_range,
    int_range as int_range,
    linear_space as linear_space,
)
from narwhals._plan.arrow.functions._repeat import (
    nulls_like as nulls_like,
    repeat as repeat,
    repeat_like as repeat_like,
    repeat_unchecked as repeat_unchecked,
    zeros as zeros,
)
from narwhals._plan.options import ExplodeOptions, SortOptions
from narwhals._utils import no_default
from narwhals.exceptions import ShapeError

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Collection,
        Iterable,
        Iterator,
        Mapping,
        Sequence,
    )

    from typing_extensions import Self, TypeAlias, Unpack

    from narwhals._arrow.typing import Incomplete
    from narwhals._plan.arrow.typing import (
        Array,
        ArrayAny,
        Arrow,
        ArrowAny,
        ArrowListT,
        ArrowT,
        BooleanScalar,
        ChunkedArray,
        ChunkedArrayAny,
        ChunkedList,
        ChunkedOrArray,
        ChunkedOrArrayAny,
        ChunkedOrArrayT,
        ChunkedOrScalar,
        ChunkedOrScalarAny,
        ChunkedOrScalarT,
        DataTypeT,
        ListArray,
        ListScalar,
        ListTypeT,
        NativeScalar,
        NonListTypeT,
        NumericScalar,
        Predicate,
        SameArrowT,
        Scalar,
        ScalarAny,
        StringScalar,
        StringType,
        UnaryFunction,
        UnaryNumeric,
        VectorFunction,
    )
    from narwhals._plan.options import RankOptions, SortMultipleOptions
    from narwhals._plan.typing import Seq
    from narwhals._typing import NoDefault
    from narwhals.typing import FillNullStrategy, NonNestedLiteral


EMPTY: Final = ""
"""The empty string."""


abs_ = t.cast("UnaryNumeric", pc.abs)
exp = t.cast("UnaryNumeric", pc.exp)
sqrt = t.cast("UnaryNumeric", pc.sqrt)
ceil = t.cast("UnaryNumeric", pc.ceil)
floor = t.cast("UnaryNumeric", pc.floor)


class ExplodeBuilder:
    """Tools for exploding lists.

    The complexity of these operations increases with:
    - Needing to preserve null/empty elements
      - All variants are cheaper if this can be skipped
    - Exploding in the context of a table
      - Where a single column is much simpler than multiple
    """

    options: ExplodeOptions

    def __init__(self, *, empty_as_null: bool = True, keep_nulls: bool = True) -> None:
        self.options = ExplodeOptions(empty_as_null=empty_as_null, keep_nulls=keep_nulls)

    @classmethod
    def from_options(cls, options: ExplodeOptions, /) -> Self:
        obj = cls.__new__(cls)
        obj.options = options
        return obj

    @t.overload
    def explode(
        self, native: ChunkedList[DataTypeT] | ListScalar[DataTypeT]
    ) -> ChunkedArray[Scalar[DataTypeT]]: ...
    @t.overload
    def explode(self, native: ListArray[DataTypeT]) -> Array[Scalar[DataTypeT]]: ...
    @t.overload
    def explode(
        self, native: Arrow[ListScalar[DataTypeT]]
    ) -> ChunkedOrArray[Scalar[DataTypeT]]: ...
    def explode(
        self, native: Arrow[ListScalar[DataTypeT]]
    ) -> ChunkedOrArray[Scalar[DataTypeT]]:
        """Explode list elements, expanding one-level into a new array.

        Equivalent to `polars.{Expr,Series}.explode`.
        """
        safe = self._fill_with_null(native) if self.options.any() else native
        if not isinstance(safe, pa.Scalar):
            return _list_explode(safe)
        return chunked_array(_list_explode(safe))

    def explode_with_indices(self, native: ChunkedList | ListArray) -> pa.Table:
        """Explode list elements, expanding one-level into a table indexing the origin.

        Returns a 2-column table, with names `"idx"` and `"values"`:

            >>> from narwhals._plan.arrow import functions as fn
            >>>
            >>> arr = fn.array([[1, 2, 3], None, [4, 5, 6], []])
            >>> fn.ExplodeBuilder().explode_with_indices(arr).to_pydict()
            {'idx': [0, 0, 0, 1, 2, 2, 2, 3], 'values': [1, 2, 3, None, 4, 5, 6, None]}
            # ^ Which sublist values come from ^ The exploded values themselves
        """
        safe = self._fill_with_null(native) if self.options.any() else native
        arrays = [_list_parent_indices(safe), _list_explode(safe)]
        return concat_horizontal(arrays, ["idx", "values"])

    def explode_column(self, native: pa.Table, column_name: str, /) -> pa.Table:
        """Explode a list-typed column in the context of `native`."""
        ca = native.column(column_name)
        if native.num_columns == 1:
            return native.from_arrays([self.explode(ca)], [column_name])
        safe = self._fill_with_null(ca) if self.options.any() else ca
        exploded = _list_explode(safe)
        col_idx = native.schema.get_field_index(column_name)
        if len(exploded) == len(native):
            return native.set_column(col_idx, column_name, exploded)
        return (
            native.remove_column(col_idx)
            .take(_list_parent_indices(safe))
            .add_column(col_idx, column_name, exploded)
        )

    def explode_columns(self, native: pa.Table, subset: Collection[str], /) -> pa.Table:
        """Explode multiple list-typed columns in the context of `native`."""
        subset = list(subset)
        arrays = native.select(subset).columns
        first = arrays[0]
        first_len = list_len(first)
        if self.options.any():
            mask = self._predicate(first_len)
            first_safe = self._fill_with_null(first, mask)
            it = (
                _list_explode(self._fill_with_null(arr, mask))
                for arr in self._iter_ensure_shape(first_len, arrays[1:])
            )
        else:
            first_safe = first
            it = (
                _list_explode(arr)
                for arr in self._iter_ensure_shape(first_len, arrays[1:])
            )
        column_names = native.column_names
        result = native
        first_result = _list_explode(first_safe)
        if len(first_result) == len(native):
            # fastpath for all length-1 lists
            # if only the first is length-1, then the others raise during iteration on either branch
            for name, arr in zip(subset, chain([first_result], it)):
                result = result.set_column(column_names.index(name), name, arr)
        else:
            result = result.drop_columns(subset).take(_list_parent_indices(first_safe))
            for name, arr in zip(subset, chain([first_result], it)):
                result = result.append_column(name, arr)
            result = result.select(column_names)
        return result

    @classmethod
    def explode_column_fast(cls, native: pa.Table, column_name: str, /) -> pa.Table:
        """Explode a list-typed column in the context of `native`, ignoring empty and nulls."""
        return cls(empty_as_null=False, keep_nulls=False).explode_column(
            native, column_name
        )

    def _iter_ensure_shape(
        self,
        first_len: ChunkedArray[pa.UInt32Scalar],
        arrays: Iterable[ChunkedArrayAny],
        /,
    ) -> Iterator[ChunkedArrayAny]:
        for arr in arrays:
            if not first_len.equals(list_len(arr)):
                msg = "exploded columns must have matching element counts"
                raise ShapeError(msg)
            yield arr

    def _predicate(self, lengths: ArrowAny, /) -> Arrow[pa.BooleanScalar]:
        """Return True for each sublist length that indicates the original sublist should be replaced with `[None]`."""
        empty_as_null, keep_nulls = self.options.empty_as_null, self.options.keep_nulls
        if empty_as_null and keep_nulls:
            return or_(is_null(lengths), eq(lengths, lit(0)))
        if empty_as_null:
            return eq(lengths, lit(0))
        return is_null(lengths)

    def _fill_with_null(
        self, native: ArrowListT, mask: Arrow[BooleanScalar] | NoDefault = no_default
    ) -> ArrowListT:
        """Replace each sublist in `native` with `[None]`, according to `self.options`.

        Arguments:
            native: List-typed arrow data.
            mask: An optional, pre-computed replacement mask. By default, this is generated from `native`.
        """
        predicate = self._predicate(list_len(native)) if mask is no_default else mask
        result: ArrowListT = when_then(predicate, lit([None], native.type), native)
        return result


def implode(native: Arrow[Scalar[DataTypeT]]) -> pa.ListScalar[DataTypeT]:
    """Aggregate values into a list.

    The returned list itself is a scalar value of `list` dtype.
    """
    arr = array(native)
    return pa.ListArray.from_arrays([0, len(arr)], arr)[0]


@t.overload
def _list_explode(native: ChunkedList[DataTypeT]) -> ChunkedArray[Scalar[DataTypeT]]: ...
@t.overload
def _list_explode(
    native: ListArray[NonListTypeT] | ListScalar[NonListTypeT],
) -> Array[Scalar[NonListTypeT]]: ...
@t.overload
def _list_explode(native: ListArray[DataTypeT]) -> Array[Scalar[DataTypeT]]: ...
@t.overload
def _list_explode(native: ListScalar[ListTypeT]) -> ListArray[ListTypeT]: ...
def _list_explode(native: Arrow[ListScalar]) -> ChunkedOrArrayAny:
    result: ChunkedOrArrayAny = pc.call_function("list_flatten", [native])
    return result


@t.overload
def _list_parent_indices(native: ChunkedList) -> ChunkedArray[pa.Int64Scalar]: ...
@t.overload
def _list_parent_indices(native: ListArray) -> pa.Int64Array: ...
def _list_parent_indices(
    native: ChunkedOrArray[ListScalar],
) -> ChunkedOrArray[pa.Int64Scalar]:
    """Don't use this withut handling nulls!"""
    result: ChunkedOrArray[pa.Int64Scalar] = pc.call_function(
        "list_parent_indices", [native]
    )
    return result


@t.overload
def list_len(native: ChunkedList) -> ChunkedArray[pa.UInt32Scalar]: ...
@t.overload
def list_len(native: ListArray) -> pa.UInt32Array: ...
@t.overload
def list_len(native: ListScalar) -> pa.UInt32Scalar: ...
@t.overload
def list_len(native: ChunkedOrScalar[ListScalar]) -> ChunkedOrScalar[pa.UInt32Scalar]: ...
@t.overload
def list_len(native: Arrow[ListScalar[Any]]) -> Arrow[pa.UInt32Scalar]: ...
def list_len(native: ArrowAny) -> ArrowAny:
    length: Incomplete = pc.list_value_length
    result: ArrowAny = length(native).cast(pa.uint32())
    return result


@t.overload
def list_get(
    native: ChunkedList[DataTypeT], index: int
) -> ChunkedArray[Scalar[DataTypeT]]: ...
@t.overload
def list_get(native: ListArray[DataTypeT], index: int) -> Array[Scalar[DataTypeT]]: ...
@t.overload
def list_get(native: ListScalar[DataTypeT], index: int) -> Scalar[DataTypeT]: ...
@t.overload
def list_get(native: SameArrowT, index: int) -> SameArrowT: ...
@t.overload
def list_get(native: ChunkedOrScalarAny, index: int) -> ChunkedOrScalarAny: ...
def list_get(native: ArrowAny, index: int) -> ArrowAny:
    list_get_: Incomplete = pc.list_element
    result: ArrowAny = list_get_(native, index)
    return result


_list_join = t.cast(
    "Callable[[ChunkedOrArrayAny, Arrow[StringScalar] | str], ChunkedArray[StringScalar] | pa.StringArray]",
    pc.binary_join,
)


# NOTE: Raised for native null-handling (https://github.com/apache/arrow/issues/48477)
@t.overload
def list_join(
    native: ChunkedList[StringType],
    separator: Arrow[StringScalar] | str,
    *,
    ignore_nulls: bool = ...,
) -> ChunkedArray[StringScalar]: ...
@t.overload
def list_join(
    native: ListArray[StringType],
    separator: Arrow[StringScalar] | str,
    *,
    ignore_nulls: bool = ...,
) -> pa.StringArray: ...
@t.overload
def list_join(
    native: ChunkedOrArray[ListScalar[StringType]],
    separator: str,
    *,
    ignore_nulls: bool = ...,
) -> ChunkedOrArray[StringScalar]: ...
def list_join(
    native: ChunkedOrArrayAny,
    separator: Arrow[StringScalar] | str,
    *,
    ignore_nulls: bool = True,
) -> ChunkedOrArrayAny:
    """Join all string items in a sublist and place a separator between them.

    Each list of values in the first input is joined using each second input as separator.
    If any input list is null or contains a null, the corresponding output will be null.
    """
    from narwhals._plan.arrow.group_by import AggSpec

    # (1): Try to return *as-is* from `pc.binary_join`
    result = _list_join(native, separator)
    if not ignore_nulls or not result.null_count:
        return result
    is_null_sensitive = pc.and_not(result.is_null(), native.is_null())
    if array(is_null_sensitive, BOOL).true_count == 0:
        return result

    # (2): Deal with only the bad kids
    lists = native.filter(is_null_sensitive)

    # (2.1): We know that `[None]` should join as `""`, and that is the only length-1 list we could have after the filter
    list_len_eq_1 = eq(list_len(lists), lit(1, UI32))
    has_a_len_1_null = any_(list_len_eq_1).as_py()
    if has_a_len_1_null:
        lists = when_then(list_len_eq_1, lit([EMPTY], lists.type), lists)

    # (2.2): Everything left falls into one of these boxes:
    # - (2.1): `[""]`
    # - (2.2): `["something", (str | None)*, None]`  <--- We fix this here and hope for the best
    # - (2.3): `[None, (None)*, None]`
    idx, v = "idx", "values"
    builder = ExplodeBuilder(empty_as_null=False, keep_nulls=False)
    explode_w_idx = builder.explode_with_indices(lists)
    implode_by_idx = AggSpec.implode(v).over(explode_w_idx.drop_null(), [idx])
    replacements = _list_join(implode_by_idx.column(v), separator)

    # (2.3): The cursed box 😨
    if len(replacements) != len(lists):
        # This is a very unlucky case to hit, because we *can* detect the issue earlier
        # but we *can't* join a table with a list in it. So we deal with the fallout now ...
        # The end result is identical to (2.1)
        indices_all = to_table(explode_w_idx.column(idx).unique(), idx)
        indices_repaired = implode_by_idx.set_column(1, v, replacements)
        replacements = (
            indices_all.join(indices_repaired, idx)
            .sort_by(idx)
            .column(v)
            .fill_null(lit(EMPTY, lists.type.value_type))
        )
    return replace_with_mask(result, is_null_sensitive, replacements)


def list_join_scalar(
    native: ListScalar[StringType],
    separator: StringScalar | str,
    *,
    ignore_nulls: bool = True,
) -> StringScalar:
    """Join all string items in a `ListScalar` and place a separator between them.

    Note:
        Consider using `list_join` or `str_join` if you don't already have `native` in this shape.
    """
    if ignore_nulls and native.is_valid:
        native = implode(_list_explode(native).drop_null())
    result: StringScalar = pc.call_function("binary_join", [native, separator])
    return result


@overload
def list_unique(native: ChunkedList) -> ChunkedList: ...
@overload
def list_unique(native: ListScalar) -> ListScalar: ...
@overload
def list_unique(native: ChunkedOrScalar[ListScalar]) -> ChunkedOrScalar[ListScalar]: ...
def list_unique(native: ChunkedOrScalar[ListScalar]) -> ChunkedOrScalar[ListScalar]:
    """Get the unique/distinct values in the list.

    There's lots of tricky stuff going on in here, but for good reasons!

    Whenever possible, we want to avoid having to deal with these pesky guys:

        [["okay", None, "still fine"], None, []]
        #                              ^^^^  ^^

    - Those kinds of list elements are ignored natively
    - `unique` is length-changing operation
    - We can't use [`pc.replace_with_mask`] on a list
    - We can't join when a table contains list columns [apache/arrow#43716]

    **But** - if we're lucky, and we got a non-awful list (or only one element) - then
    most issues vanish.

    [`pc.replace_with_mask`]: https://arrow.apache.org/docs/python/generated/pyarrow.compute.replace_with_mask.html
    [apache/arrow#43716]: https://github.com/apache/arrow/issues/43716
    """
    from narwhals._plan.arrow.group_by import AggSpec

    if isinstance(native, pa.Scalar):
        scalar = t.cast("pa.ListScalar[Any]", native)
        if scalar.is_valid and (len(scalar) > 1):
            return implode(_list_explode(native).unique())
        return scalar
    idx, v = "index", "values"
    names = idx, v
    len_not_eq_0 = not_eq(list_len(native), lit(0))
    can_fastpath = all_(len_not_eq_0, ignore_nulls=False).as_py()
    if can_fastpath:
        arrays = [_list_parent_indices(native), _list_explode(native)]
        return AggSpec.unique(v).over_index(concat_horizontal(arrays, names), idx)
    # Oh no - we caught a bad one!
    # We need to split things into good/bad - and only work on the good stuff.
    # `int_range` is acting like `parent_indices`, but doesn't give up when it see's `None` or `[]`
    indexed = concat_horizontal([int_range(len(native)), native], names)
    valid = indexed.filter(len_not_eq_0)
    invalid = indexed.filter(or_(native.is_null(), not_(len_not_eq_0)))
    # To keep track of where we started, our index needs to be exploded with the list elements
    explode_with_index = ExplodeBuilder.explode_column_fast(valid, v)
    valid_unique = AggSpec.unique(v).over(explode_with_index, [idx])
    # And now, because we can't join - we do a poor man's version of one 😉
    return concat_tables([valid_unique, invalid]).sort_by(idx).column(v)


def list_contains(
    native: ChunkedOrScalar[ListScalar], item: NonNestedLiteral | ScalarAny
) -> ChunkedOrScalar[pa.BooleanScalar]:
    from narwhals._plan.arrow.group_by import AggSpec

    if isinstance(native, pa.Scalar):
        scalar = t.cast("pa.ListScalar[Any]", native)
        if scalar.is_valid:
            if len(scalar):
                value_type = scalar.type.value_type
                return any_(eq_missing(_list_explode(scalar), lit(item).cast(value_type)))
            return lit(False, BOOL)
        return lit(None, BOOL)
    builder = ExplodeBuilder(empty_as_null=False, keep_nulls=False)
    tbl = builder.explode_with_indices(native)
    idx, name = tbl.column_names
    contains = eq_missing(tbl.column(name), item)
    l_contains = AggSpec.any(name).over_index(tbl.set_column(1, name, contains), idx)
    # Here's the really key part: this mask has the same result we want to return
    # So by filling the `True`, we can flip those to `False` if needed
    # But if we were already `None` or `False` - then that's sticky
    propagate_invalid: ChunkedArray[pa.BooleanScalar] = not_eq(list_len(native), lit(0))
    return replace_with_mask(propagate_invalid, propagate_invalid, l_contains)


def list_sort(
    native: ChunkedList, *, descending: bool = False, nulls_last: bool = False
) -> ChunkedList:
    """Sort the sublists in this column.

    Works in a similar way to `list_unique` and `list_join`.

    1. Select only sublists that require sorting (`None`, 0-length, and 1-length lists are noops)
    2. Explode -> Sort -> Implode -> Concat
    """
    from narwhals._plan.arrow.group_by import AggSpec

    idx, v = "idx", "values"
    is_not_sorted = gt(list_len(native), lit(1))
    indexed = concat_horizontal([int_range(len(native)), native], [idx, v])
    exploded = ExplodeBuilder.explode_column_fast(indexed.filter(is_not_sorted), v)
    indices = sort_indices(
        exploded, idx, v, descending=[False, descending], nulls_last=nulls_last
    )
    exploded_sorted = exploded.take(indices)
    implode_by_idx = AggSpec.implode(v).over(exploded_sorted, [idx])
    passthrough = indexed.filter(fill_null(not_(is_not_sorted), True))
    return concat_tables([implode_by_idx, passthrough]).sort_by(idx).column(v)


def list_sort_scalar(
    native: ListScalar[NonListTypeT], options: SortOptions | None = None
) -> pa.ListScalar[NonListTypeT]:
    native = t.cast("pa.ListScalar[NonListTypeT]", native)
    if native.is_valid and len(native) > 1:
        arr = _list_explode(native)
        return implode(arr.take(sort_indices(arr, options=options)))
    return native


@t.overload
def when_then(
    predicate: ChunkedArray[BooleanScalar], then: ScalarAny
) -> ChunkedArrayAny: ...
@t.overload
def when_then(predicate: Array[BooleanScalar], then: ScalarAny) -> ArrayAny: ...
@t.overload
def when_then(
    predicate: Predicate, then: SameArrowT, otherwise: SameArrowT | None
) -> SameArrowT: ...
@t.overload
def when_then(predicate: Predicate, then: ScalarAny, otherwise: ArrowT) -> ArrowT: ...
@t.overload
def when_then(
    predicate: Predicate, then: ArrowT, otherwise: ScalarAny | NonNestedLiteral = ...
) -> ArrowT: ...
@t.overload
def when_then(
    predicate: Predicate, then: ArrowAny, otherwise: ArrowAny | NonNestedLiteral = None
) -> Incomplete: ...
def when_then(
    predicate: Predicate, then: ArrowAny, otherwise: ArrowAny | NonNestedLiteral = None
) -> Incomplete:
    """Thin wrapper around `pyarrow.compute.if_else`.

    - Supports a 2-arg form, like `pl.when(...).then(...)`
    - Accepts python literals, but only in the `otherwise` position
    """
    if is_non_nested_literal(otherwise):
        otherwise = lit(otherwise, then.type)
    return pc.if_else(predicate, then, otherwise)


def sum_(native: Incomplete) -> NativeScalar:
    return pc.sum(native, min_count=0)


def first(native: ChunkedOrArrayAny) -> NativeScalar:
    return pc.first(native, options=pa_options.scalar_aggregate())


def last(native: ChunkedOrArrayAny) -> NativeScalar:
    return pc.last(native, options=pa_options.scalar_aggregate())


min_ = pc.min
# TODO @dangotbanned: Wrap horizontal functions with correct typing
# Should only return scalar if all elements are as well
min_horizontal = pc.min_element_wise
max_ = pc.max
max_horizontal = pc.max_element_wise
mean = t.cast("Callable[[ChunkedOrArray[pc.NumericScalar]], pa.DoubleScalar]", pc.mean)
count = pc.count
median = pc.approximate_median
std = pc.stddev
var = pc.variance
quantile = pc.quantile


def mode_all(native: ChunkedArrayAny) -> ChunkedArrayAny:
    struct = pc.mode(native, n=len(native))
    indices: pa.Int32Array = struct.field("count").dictionary_encode().indices  # type: ignore[attr-defined]
    index_true_modes = lit(0)
    return chunked_array(struct.field("mode").filter(pc.equal(indices, index_true_modes)))


def mode_any(native: ChunkedArrayAny) -> NativeScalar:
    return first(pc.mode(native, n=1).field("mode"))


def kurtosis_skew(
    native: ChunkedArray[pc.NumericScalar], function: Literal["kurtosis", "skew"], /
) -> NativeScalar:
    result: NativeScalar
    if compat.HAS_KURTOSIS_SKEW:
        if pa.types.is_null(native.type):
            native = native.cast(F64)
        result = getattr(pc, function)(native)
    else:
        non_null = native.drop_null()
        if len(non_null) == 0:
            result = lit(None, F64)
        elif len(non_null) == 1:
            result = lit(float("nan"))
        elif function == "skew" and len(non_null) == 2:
            result = lit(0.0, F64)
        else:
            m = sub(non_null, mean(non_null))
            m2 = mean(power(m, lit(2)))
            if function == "kurtosis":
                m4 = mean(power(m, lit(4)))
                result = sub(pc.divide(m4, power(m2, lit(2))), lit(3))
            else:
                m3 = mean(power(m, lit(3)))
                result = pc.divide(m3, power(m2, lit(1.5)))
    return result


def clip_lower(
    native: ChunkedOrScalarAny, lower: ChunkedOrScalarAny
) -> ChunkedOrScalarAny:
    return max_horizontal(native, lower)


def clip_upper(
    native: ChunkedOrScalarAny, upper: ChunkedOrScalarAny
) -> ChunkedOrScalarAny:
    return min_horizontal(native, upper)


def clip(
    native: ChunkedOrScalarAny, lower: ChunkedOrScalarAny, upper: ChunkedOrScalarAny
) -> ChunkedOrScalarAny:
    return clip_lower(clip_upper(native, upper), lower)


def n_unique(native: Any) -> pa.Int64Scalar:
    return count(native, mode="all")


@t.overload
def round(native: ChunkedOrScalarAny, decimals: int = ...) -> ChunkedOrScalarAny: ...
@t.overload
def round(native: ChunkedOrArrayT, decimals: int = ...) -> ChunkedOrArrayT: ...
def round(native: ArrowAny, decimals: int = 0) -> ArrowAny:
    return pc.round(native, decimals, round_mode="half_towards_infinity")


def log(native: ChunkedOrScalarAny, base: float = math.e) -> ChunkedOrScalarAny:
    return t.cast("ChunkedOrScalarAny", pc.logb(native, lit(base)))


def diff(native: ChunkedOrArrayT, n: int = 1) -> ChunkedOrArrayT:
    # pyarrow.lib.ArrowInvalid: Vector kernel cannot execute chunkwise and no chunked exec function was defined
    return (
        pc.pairwise_diff(native, n)
        if isinstance(native, pa.Array)
        else chunked_array(pc.pairwise_diff(native.combine_chunks(), n))
    )


def shift(
    native: ChunkedArrayAny, n: int, *, fill_value: NonNestedLiteral = None
) -> ChunkedArrayAny:
    if n == 0:
        return native
    arr = native
    if n > 0:
        filled = repeat_like(fill_value, n, arr)
        arrays = [filled, *arr.slice(length=arr.length() - n).chunks]
    else:
        filled = repeat_like(fill_value, -n, arr)
        arrays = [*arr.slice(offset=-n).chunks, filled]
    return pa.chunked_array(arrays)


def rank(native: ChunkedArrayAny, rank_options: RankOptions) -> ChunkedArrayAny:
    arr = native if compat.RANK_ACCEPTS_CHUNKED else array(native)
    if rank_options.method == "average":
        # Adapted from https://github.com/pandas-dev/pandas/blob/f4851e500a43125d505db64e548af0355227714b/pandas/core/arrays/arrow/array.py#L2290-L2316
        order = pa_options.ORDER[rank_options.descending]
        min = preserve_nulls(arr, pc.rank(arr, order, tiebreaker="min").cast(F64))
        max = pc.rank(arr, order, tiebreaker="max").cast(F64)
        ranked = pc.divide(pc.add(min, max), lit(2, F64))
    else:
        ranked = preserve_nulls(native, pc.rank(arr, options=rank_options.to_arrow()))
    return chunked_array(ranked)


def null_count(native: ChunkedOrArrayAny) -> pa.Int64Scalar:
    return pc.count(native, mode="only_null")


def preserve_nulls(
    before: ChunkedOrArrayAny, after: ChunkedOrArrayT, /
) -> ChunkedOrArrayT:
    return when_then(is_not_null(before), after) if before.null_count else after


drop_nulls = t.cast("VectorFunction[...]", pc.drop_null)


_FILL_NULL_STRATEGY: Mapping[FillNullStrategy, UnaryFunction] = {
    "forward": pc.fill_null_forward,
    "backward": pc.fill_null_backward,
}


def _fill_null_forward_limit(native: ChunkedArrayAny, limit: int) -> ChunkedArrayAny:
    SENTINEL = lit(-1)  # noqa: N806
    is_not_null = native.is_valid()
    index = int_range(len(native), chunked=False)
    index_not_null = cum_max(when_then(is_not_null, index, SENTINEL))
    # NOTE: The correction here is for nulls at either end of the array
    # They should be preserved when the `strategy` would need an out-of-bounds index
    not_oob = not_eq(index_not_null, SENTINEL)
    index_not_null = when_then(not_oob, index_not_null)
    beyond_limit = gt(sub(index, index_not_null), lit(limit))
    return when_then(or_(is_not_null, beyond_limit), native, native.take(index_not_null))


@t.overload
def fill_null(
    native: ChunkedOrScalarT, value: NonNestedLiteral | ArrowAny
) -> ChunkedOrScalarT: ...
@t.overload
def fill_null(
    native: ChunkedOrArrayT, value: ScalarAny | NonNestedLiteral | ChunkedOrArrayT
) -> ChunkedOrArrayT: ...
@t.overload
def fill_null(
    native: ChunkedOrScalarAny, value: ChunkedOrScalarAny | NonNestedLiteral
) -> ChunkedOrScalarAny: ...
def fill_null(native: ArrowAny, value: ArrowAny | NonNestedLiteral) -> ArrowAny:
    fill_value: Incomplete = value
    result: ArrowAny = pc.fill_null(native, fill_value)
    return result


@t.overload
def fill_nan(
    native: ChunkedOrScalarT, value: NonNestedLiteral | ArrowAny
) -> ChunkedOrScalarT: ...
@t.overload
def fill_nan(native: SameArrowT, value: NonNestedLiteral | ArrowAny) -> SameArrowT: ...
def fill_nan(native: ArrowAny, value: NonNestedLiteral | ArrowAny) -> Incomplete:
    return when_then(is_not_nan(native), native, value)


def fill_null_forward(native: ChunkedArrayAny) -> ChunkedArrayAny:
    return fill_null_with_strategy(native, "forward")


def fill_null_with_strategy(
    native: ChunkedArrayAny, strategy: FillNullStrategy, limit: int | None = None
) -> ChunkedArrayAny:
    null_count = native.null_count
    if null_count == 0 or (null_count == len(native)):
        return native
    if limit is None:
        return _FILL_NULL_STRATEGY[strategy](native)
    if strategy == "forward":
        return _fill_null_forward_limit(native, limit)
    return reverse(_fill_null_forward_limit(reverse(native), limit))


def _ensure_all_replaced(
    native: ChunkedOrScalarAny, unmatched: ArrowAny
) -> ValueError | None:
    if not any_(unmatched).as_py():
        return None
    msg = (
        "replace_strict did not replace all non-null values.\n\n"
        f"The following did not get replaced: {chunked_array(native).filter(array(unmatched)).unique().to_pylist()}"
    )
    return ValueError(msg)


def replace_strict(
    native: ChunkedOrScalarAny,
    old: Seq[Any],
    new: Seq[Any],
    dtype: pa.DataType | None = None,
) -> ChunkedOrScalarAny:
    if isinstance(native, pa.Scalar):
        idxs: ArrayAny = array(pc.index_in(native, pa.array(old)))
        result: ChunkedOrScalarAny = pa.array(new).take(idxs)[0]
    else:
        idxs = pc.index_in(native, pa.array(old))
        result = chunked_array(pa.array(new).take(idxs))
    if err := _ensure_all_replaced(native, and_(is_not_null(native), is_null(idxs))):
        raise err
    return result.cast(dtype) if dtype else result


def replace_strict_default(
    native: ChunkedOrScalarAny,
    old: Seq[Any],
    new: Seq[Any],
    default: ChunkedOrScalarAny,
    dtype: pa.DataType | None = None,
) -> ChunkedOrScalarAny:
    idxs = pc.index_in(native, pa.array(old))
    result = pa.array(new).take(array(idxs))
    result = when_then(is_null(idxs), default, result.cast(dtype) if dtype else result)
    return chunked_array(result) if isinstance(native, pa.ChunkedArray) else result[0]


@overload
def replace_with_mask(
    native: ChunkedOrArrayT, mask: Predicate, replacements: ChunkedOrArrayAny
) -> ChunkedOrArrayT: ...
@overload
def replace_with_mask(
    native: ChunkedOrArrayAny, mask: Predicate, replacements: ChunkedOrArrayAny
) -> ChunkedOrArrayAny: ...
def replace_with_mask(
    native: ChunkedOrArrayAny, mask: Predicate, replacements: ChunkedOrArrayAny
) -> ChunkedOrArrayAny:
    """Replace elements of `native`, at positions defined by `mask`.

    The length of `replacements` must equal the number of `True` values in `mask`.
    """
    if isinstance(native, pa.ChunkedArray):
        args = [array(p) for p in (native, mask, replacements)]
        return chunked_array(pc.call_function("replace_with_mask", args))
    args = [native, array(mask), array(replacements)]
    result: ChunkedOrArrayAny = pc.call_function("replace_with_mask", args)
    return result


def random_indices(
    end: int, /, n: int, *, with_replacement: bool = False, seed: int | None = None
) -> ArrayAny:
    """Generate `n` random indices within the range `[0, end)`."""
    # NOTE: Review this path if anything changes upstream
    # https://github.com/apache/arrow/issues/47288#issuecomment-3597653670
    if with_replacement:
        rand_values = pc.random(n, initializer="system" if seed is None else seed)
        return round(multiply(rand_values, lit(end - 1))).cast(I64)

    import numpy as np  # ignore-banned-import

    return array(np.random.default_rng(seed).choice(np.arange(end), n, replace=False))


@overload
def sort_indices(
    native: ChunkedOrArrayAny, *, options: SortOptions | None
) -> pa.UInt64Array: ...
@overload
def sort_indices(
    native: ChunkedOrArrayAny, *, descending: bool = ..., nulls_last: bool = ...
) -> pa.UInt64Array: ...
@overload
def sort_indices(
    native: pa.Table,
    *by: Unpack[tuple[str, Unpack[tuple[str, ...]]]],
    options: SortOptions | SortMultipleOptions | None,
) -> pa.UInt64Array: ...
@overload
def sort_indices(
    native: pa.Table,
    *by: Unpack[tuple[str, Unpack[tuple[str, ...]]]],
    descending: bool | Sequence[bool] = ...,
    nulls_last: bool = ...,
) -> pa.UInt64Array: ...
def sort_indices(
    native: ChunkedOrArrayAny | pa.Table,
    *by: str,
    options: SortOptions | SortMultipleOptions | None = None,
    descending: bool | Sequence[bool] = False,
    nulls_last: bool = False,
) -> pa.UInt64Array:
    """Return the indices that would sort an array or table.

    Arguments:
        native: Any non-scalar arrow data.
        *by: Column(s) to sort by. Only applicable to `Table` and must use at least one name.
        options: An *already-parsed* options instance.
            **Has higher precedence** than `descending` and `nulls_last`.
        descending: Sort in descending order. When sorting by multiple columns,
            can be specified per column by passing a sequence of booleans.
        nulls_last: Place null values last.

    Notes:
        Most commonly used as input for `take`, which forms a `sort_by` operation.
    """
    if not isinstance(native, pa.Table):
        if options:
            descending = options.descending
            nulls_last = options._ensure_single_nulls_last("pyarrow")
        a_opts = pa_options.array_sort(descending=descending, nulls_last=nulls_last)
        return pc.array_sort_indices(native, options=a_opts)
    opts = (
        options.to_arrow(by)
        if options
        else pa_options.sort(*by, descending=descending, nulls_last=nulls_last)
    )
    return pc.sort_indices(native, options=opts)


def unsort_indices(indices: pa.UInt64Array, /) -> pa.Int64Array:
    """Return the inverse permutation of the given indices.

    Arguments:
        indices: The output of `sort_indices`.

    Examples:
        We can use this pair of functions to recreate a windowed `pl.row_index`

        >>> import polars as pl
        >>> data = {"by": [5, 2, 5, None]}
        >>> df = pl.DataFrame(data)
        >>> df.select(
        ...     pl.row_index().over(order_by="by", descending=True, nulls_last=False)
        ... ).to_series().to_list()
        [1, 3, 2, 0]

        Now in `pyarrow`

        >>> import pyarrow as pa
        >>> from narwhals._plan.arrow.functions import sort_indices, unsort_indices
        >>> df = pa.Table.from_pydict(data)
        >>> unsort_indices(
        ...     sort_indices(df, "by", descending=True, nulls_last=False)
        ... ).to_pylist()
        [1, 3, 2, 0]
    """
    return (
        pc.inverse_permutation(indices.cast(pa.int64()))  # type: ignore[attr-defined]
        if compat.HAS_SCATTER
        else int_range(len(indices), chunked=False).take(pc.sort_indices(indices))
    )


SearchSortedSide: TypeAlias = Literal["left", "right"]


# NOTE @dangotbanned: (wish) replacing `np.searchsorted`?
@t.overload
def search_sorted(
    native: ChunkedOrArrayT,
    element: ChunkedOrArray[NumericScalar] | Sequence[float],
    *,
    side: SearchSortedSide = ...,
) -> ChunkedOrArrayT: ...
# NOTE: scalar case may work with only `partition_nth_indices`?
@t.overload
def search_sorted(
    native: ChunkedOrArrayT, element: float, *, side: SearchSortedSide = ...
) -> ScalarAny: ...
def search_sorted(
    native: ChunkedOrArrayT,
    element: ChunkedOrArray[NumericScalar] | Sequence[float] | float,
    *,
    side: SearchSortedSide = "left",
) -> ChunkedOrArrayT | ScalarAny:
    """Find indices where elements should be inserted to maintain order."""
    import numpy as np  # ignore-banned-import

    indices = np.searchsorted(element, native, side=side)
    if isinstance(indices, np.generic):
        return lit(indices)
    if isinstance(native, pa.ChunkedArray):
        return chunked_array([indices])
    return array(indices)


def hist_bins(
    native: ChunkedArrayAny,
    bins: Sequence[float] | ChunkedArray[NumericScalar],
    *,
    include_breakpoint: bool,
) -> Mapping[str, Iterable[Any]]:
    """Bin values into buckets and count their occurrences.

    Notes:
        Assumes that the following edge cases have been handled:
        - `len(bins) >= 2`
        - `bins` increase monotonically
        - `bin[0] != bin[-1]`
        - `native` contains values that are non-null (including NaN)
    """
    if len(bins) == 2:
        upper = bins[1]
        count = array(is_between(native, bins[0], upper, closed="both"), BOOL).true_count
        if include_breakpoint:
            return {"breakpoint": [upper], "count": [count]}
        return {"count": [count]}

    # lowest bin is inclusive
    # NOTE: `np.unique` behavior sorts first
    value_counts = (
        when_then(not_eq(native, lit(bins[0])), search_sorted(native, bins), 1)
        .sort()
        .value_counts()
    )
    values, counts = struct.fields(value_counts, "values", "counts")
    bin_count = len(bins)
    int_range_ = int_range(1, bin_count, chunked=False)
    mask = is_in(int_range_, values)
    replacements = counts.filter(is_in(values, int_range_))
    counts = replace_with_mask(zeros(bin_count - 1), mask, replacements)

    if include_breakpoint:
        return {"breakpoint": bins[1:], "count": counts}
    return {"count": counts}


def hist_zeroed_data(
    arg: int | Sequence[float], *, include_breakpoint: bool
) -> Mapping[str, Iterable[Any]]:
    # NOTE: If adding `linear_space` and `zeros` to `CompliantNamespace`, consider moving this.
    n = arg if isinstance(arg, int) else len(arg) - 1
    if not include_breakpoint:
        return {"count": zeros(n)}
    bp = linear_space(0, 1, arg, closed="right") if isinstance(arg, int) else arg[1:]
    return {"breakpoint": bp, "count": zeros(n)}
