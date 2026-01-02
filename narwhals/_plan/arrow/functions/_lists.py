"""List namespace functions."""

from __future__ import annotations

import builtins
import typing as t
from itertools import chain
from typing import TYPE_CHECKING, Any, Final, overload

import pyarrow as pa  # ignore-banned-import
import pyarrow.compute as pc  # ignore-banned-import

from narwhals._plan.arrow.functions._aggregation import implode
from narwhals._plan.arrow.functions._bin_op import eq, gt, not_eq, or_
from narwhals._plan.arrow.functions._boolean import all, any, eq_missing, is_null, not_
from narwhals._plan.arrow.functions._construction import (
    array,
    chunked_array,
    concat_horizontal,
    concat_tables,
    lit,
    to_table,
)
from narwhals._plan.arrow.functions._dtypes import BOOL, U32
from narwhals._plan.arrow.functions._multiplex import (
    fill_null,
    replace_with_mask,
    when_then,
)
from narwhals._plan.arrow.functions._ranges import int_range
from narwhals._plan.arrow.functions._sort import sort_indices
from narwhals._plan.options import ExplodeOptions, SortOptions
from narwhals._utils import no_default
from narwhals.exceptions import ShapeError

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterable, Iterator

    from typing_extensions import Self

    from narwhals._plan.arrow.typing import (
        Array,
        Arrow,
        ArrowAny,
        ArrowListT,
        BooleanScalar,
        ChunkedArray,
        ChunkedArrayAny,
        ChunkedList,
        ChunkedOrArray,
        ChunkedOrArrayAny,
        ChunkedOrScalar,
        ChunkedOrScalarAny,
        DataTypeT,
        ListArray,
        ListScalar,
        ListTypeT,
        NonListTypeT,
        SameArrowT,
        Scalar,
        ScalarAny,
        StringScalar,
        StringType,
    )
    from narwhals._typing import NoDefault
    from narwhals.typing import NonNestedLiteral


__all__ = [
    "ExplodeBuilder",
    "contains",
    "get",
    "join",
    "join_scalar",
    "len",
    "sort",
    "sort_scalar",
    "unique",
]


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

    @overload
    def explode(
        self, native: ChunkedList[DataTypeT] | ListScalar[DataTypeT]
    ) -> ChunkedArray[Scalar[DataTypeT]]: ...
    @overload
    def explode(self, native: ListArray[DataTypeT]) -> Array[Scalar[DataTypeT]]: ...
    @overload
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
            return _explode(safe)
        return chunked_array(_explode(safe))

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
        arrays = [_list_parent_indices(safe), _explode(safe)]
        return concat_horizontal(arrays, ["idx", "values"])

    def explode_column(self, native: pa.Table, column_name: str, /) -> pa.Table:
        """Explode a list-typed column in the context of `native`."""
        ca = native.column(column_name)
        if native.num_columns == 1:
            return native.from_arrays([self.explode(ca)], [column_name])
        safe = self._fill_with_null(ca) if self.options.any() else ca
        exploded = _explode(safe)
        col_idx = native.schema.get_field_index(column_name)
        if exploded.length() == native.num_rows:
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
        first_len = len(first)
        if self.options.any():
            mask = self._predicate(first_len)
            first_safe = self._fill_with_null(first, mask)
            it = (
                _explode(self._fill_with_null(arr, mask))
                for arr in self._iter_ensure_shape(first_len, arrays[1:])
            )
        else:
            first_safe = first
            it = (_explode(arr) for arr in self._iter_ensure_shape(first_len, arrays[1:]))
        column_names = native.column_names
        result = native
        first_result = _explode(first_safe)
        if first_result.length() == native.num_rows:
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
            if not first_len.equals(len(arr)):
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
        predicate = self._predicate(len(native)) if mask is no_default else mask
        result: ArrowListT = when_then(predicate, lit([None], native.type), native)
        return result


@overload
def len(native: ChunkedList) -> ChunkedArray[pa.UInt32Scalar]: ...
@overload
def len(native: ListArray) -> pa.UInt32Array: ...
@overload
def len(native: ListScalar) -> pa.UInt32Scalar: ...
@overload
def len(native: ChunkedOrScalar[ListScalar]) -> ChunkedOrScalar[pa.UInt32Scalar]: ...
@overload
def len(native: Arrow[ListScalar[Any]]) -> Arrow[pa.UInt32Scalar]: ...
def len(native: ArrowAny) -> ArrowAny:
    """Return the number of elements in each sublist.

    Null values count towards the total.

    Arguments:
        native: List-typed arrow data.

    Important:
        This is **not** [`builtins.len`]!

    [`builtins.len`]: https://docs.python.org/3/library/functions.html#len
    """
    result: ArrowAny = pc.call_function("list_value_length", [native]).cast(U32)
    return result


@overload
def get(
    native: ChunkedList[DataTypeT], index: int
) -> ChunkedArray[Scalar[DataTypeT]]: ...
@overload
def get(native: ListArray[DataTypeT], index: int) -> Array[Scalar[DataTypeT]]: ...
@overload
def get(native: ListScalar[DataTypeT], index: int) -> Scalar[DataTypeT]: ...
@overload
def get(native: SameArrowT, index: int) -> SameArrowT: ...
@overload
def get(native: ChunkedOrScalarAny, index: int) -> ChunkedOrScalarAny: ...
def get(native: ArrowAny, index: int) -> ArrowAny:
    """Get the value by index in the sublists.

    Arguments:
        native: List-typed arrow data.
        index: Index to return per sublist.
    """
    result: ArrowAny = pc.call_function("list_element", [native, index])
    return result


EMPTY: Final = ""
"""The empty string."""


# NOTE: Raised for native null-handling (https://github.com/apache/arrow/issues/48477)
@overload
def join(
    native: ChunkedList[StringType],
    separator: Arrow[StringScalar] | str,
    *,
    ignore_nulls: bool = ...,
) -> ChunkedArray[StringScalar]: ...
@overload
def join(
    native: ListArray[StringType],
    separator: Arrow[StringScalar] | str,
    *,
    ignore_nulls: bool = ...,
) -> pa.StringArray: ...
@overload
def join(
    native: ChunkedOrArray[ListScalar[StringType]],
    separator: str,
    *,
    ignore_nulls: bool = ...,
) -> ChunkedOrArray[StringScalar]: ...
def join(
    native: ChunkedOrArrayAny,
    separator: Arrow[StringScalar] | str,
    *,
    ignore_nulls: bool = True,
) -> ChunkedOrArrayAny:
    """Join all string items in a sublist and place a separator between them.

    Arguments:
        native: List-typed arrow data, where the inner type is String.
        separator: String to separate the items with
        ignore_nulls: If set to False, null values will be propagated.
            If the sub-list contains any null values, the output is None.
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
    list_len_eq_1 = eq(len(lists), lit(1, U32))
    has_a_len_1_null = any(list_len_eq_1).as_py()
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
    if builtins.len(replacements) != builtins.len(lists):
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


def join_scalar(
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
        native = implode(_explode(native).drop_null())
    result: StringScalar = pc.call_function("binary_join", [native, separator])
    return result


@overload
def unique(native: ChunkedList) -> ChunkedList: ...
@overload
def unique(native: ListScalar) -> ListScalar: ...
@overload
def unique(native: ChunkedOrScalar[ListScalar]) -> ChunkedOrScalar[ListScalar]: ...
def unique(native: ChunkedOrScalar[ListScalar]) -> ChunkedOrScalar[ListScalar]:
    """Get the distinct values in each sublist.

    Arguments:
        native: List-typed arrow data.

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
        scalar = _typing_list_scalar(native)
        if scalar.is_valid and (builtins.len(scalar) > 1):
            return implode(_explode(native).unique())
        return scalar
    idx, v = "index", "values"
    names = idx, v
    len_not_eq_0 = not_eq(len(native), lit(0))
    can_fastpath = all(len_not_eq_0, ignore_nulls=False).as_py()
    if can_fastpath:
        arrays = [_list_parent_indices(native), _explode(native)]
        return AggSpec.unique(v).over_index(concat_horizontal(arrays, names), idx)
    # Oh no - we caught a bad one!
    # We need to split things into good/bad - and only work on the good stuff.
    # `int_range` is acting like `parent_indices`, but doesn't give up when it see's `None` or `[]`
    indexed = concat_horizontal([int_range(native.length()), native], names)
    valid = indexed.filter(len_not_eq_0)
    invalid = indexed.filter(or_(native.is_null(), not_(len_not_eq_0)))
    # To keep track of where we started, our index needs to be exploded with the list elements
    explode_with_index = ExplodeBuilder.explode_column_fast(valid, v)
    valid_unique = AggSpec.unique(v).over(explode_with_index, [idx])
    # And now, because we can't join - we do a poor man's version of one 😉
    return concat_tables([valid_unique, invalid]).sort_by(idx).column(v)


def contains(
    native: ChunkedOrScalar[ListScalar], item: NonNestedLiteral | ScalarAny
) -> ChunkedOrScalar[pa.BooleanScalar]:
    """Check if sublists contain the given item.

    Arguments:
        native: List-typed arrow data.
        item: Item that will be checked for membership
    """
    from narwhals._plan.arrow.group_by import AggSpec

    if isinstance(native, pa.Scalar):
        scalar = _typing_list_scalar(native)
        if scalar.is_valid:
            if builtins.len(scalar):
                value_type = scalar.type.value_type
                return any(eq_missing(_explode(scalar), lit(item).cast(value_type)))
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
    propagate_invalid: ChunkedArray[pa.BooleanScalar] = not_eq(len(native), lit(0))
    return replace_with_mask(propagate_invalid, propagate_invalid, l_contains)


def sort(
    native: ChunkedList, *, descending: bool = False, nulls_last: bool = False
) -> ChunkedList:
    """Sort the sublists in this column.

    Works in a similar way to `list_unique` and `list_join`.

    1. Select only sublists that require sorting (`None`, 0-length, and 1-length lists are noops)
    2. Explode -> Sort -> Implode -> Concat
    """
    from narwhals._plan.arrow.group_by import AggSpec

    idx, v = "idx", "values"
    is_not_sorted = gt(len(native), lit(1))
    indexed = concat_horizontal([int_range(native.length()), native], [idx, v])
    exploded = ExplodeBuilder.explode_column_fast(indexed.filter(is_not_sorted), v)
    indices = sort_indices(
        exploded, idx, v, descending=[False, descending], nulls_last=nulls_last
    )
    exploded_sorted = exploded.take(indices)
    implode_by_idx = AggSpec.implode(v).over(exploded_sorted, [idx])
    passthrough = indexed.filter(fill_null(not_(is_not_sorted), True))
    return concat_tables([implode_by_idx, passthrough]).sort_by(idx).column(v)


# TODO @dangotbanned: Docstring?
def sort_scalar(
    native: ListScalar[NonListTypeT], options: SortOptions | None = None
) -> pa.ListScalar[NonListTypeT]:
    native = _typing_list_scalar(native)
    if native.is_valid and builtins.len(native) > 1:
        arr = _explode(native)
        return implode(arr.take(sort_indices(arr, options=options)))
    return native


def _typing_list_scalar(native: ListScalar[DataTypeT], /) -> pa.ListScalar[DataTypeT]:
    """**Runtime noop**.

    Just performs a useful `typing.cast`:

        pa.Scalar[pa.ListType[DataTypeT]]  # This isn't a real thing at runtime
        pa.ListScalar[DataTypeT]           # Defines: `values`, `__len__`
    """
    return t.cast("pa.ListScalar[DataTypeT]", native)


_list_join = t.cast(
    "Callable[[ChunkedOrArrayAny, Arrow[StringScalar] | str], ChunkedArray[StringScalar] | pa.StringArray]",
    pc.binary_join,
)


@overload
def _explode(native: ChunkedList[DataTypeT]) -> ChunkedArray[Scalar[DataTypeT]]: ...
@overload
def _explode(
    native: ListArray[NonListTypeT] | ListScalar[NonListTypeT],
) -> Array[Scalar[NonListTypeT]]: ...
@overload
def _explode(native: ListArray[DataTypeT]) -> Array[Scalar[DataTypeT]]: ...
@overload
def _explode(native: ListScalar[ListTypeT]) -> ListArray[ListTypeT]: ...
def _explode(native: Arrow[ListScalar]) -> ChunkedOrArrayAny:
    result: ChunkedOrArrayAny = pc.call_function("list_flatten", [native])
    return result


@overload
def _list_parent_indices(native: ChunkedList) -> ChunkedArray[pa.Int64Scalar]: ...
@overload
def _list_parent_indices(native: ListArray) -> pa.Int64Array: ...
def _list_parent_indices(
    native: ChunkedOrArray[ListScalar],
) -> ChunkedOrArray[pa.Int64Scalar]:
    result: ChunkedOrArray[pa.Int64Scalar] = pc.call_function(
        "list_parent_indices", [native]
    )
    return result
