"""Native functions, aliased and/or with behavior aligned to `polars`."""

from __future__ import annotations

import math
import typing as t
from collections.abc import Callable, Collection, Iterator, Sequence
from itertools import chain
from typing import TYPE_CHECKING, Any, Final, Literal, overload

import pyarrow as pa  # ignore-banned-import
import pyarrow.compute as pc  # ignore-banned-import

from narwhals._arrow.utils import (
    cast_for_truediv,
    chunked_array as _chunked_array,
    concat_tables as concat_tables,  # noqa: PLC0414
    floordiv_compat as _floordiv,
    narwhals_to_native_dtype as _dtype_native,
)
from narwhals._plan import common, expressions as ir
from narwhals._plan._guards import is_non_nested_literal
from narwhals._plan.arrow import compat, options as pa_options
from narwhals._plan.expressions import functions as F, operators as ops
from narwhals._plan.options import ExplodeOptions, SortOptions
from narwhals._utils import Version, no_default
from narwhals.exceptions import ShapeError

if TYPE_CHECKING:
    import datetime as dt
    from collections.abc import Iterable, Mapping

    from typing_extensions import Self, TypeAlias, TypeIs, TypeVarTuple, Unpack

    from narwhals._arrow.typing import Incomplete
    from narwhals._plan.arrow.acero import Field
    from narwhals._plan.arrow.typing import (
        Array,
        ArrayAny,
        Arrow,
        ArrowAny,
        ArrowListT,
        ArrowT,
        BinaryComp,
        BinaryFunction,
        BinaryLogical,
        BinaryNumericTemporal,
        BinOp,
        BooleanLengthPreserving,
        BooleanScalar,
        BoolType,
        ChunkedArray,
        ChunkedArrayAny,
        ChunkedList,
        ChunkedOrArray,
        ChunkedOrArrayAny,
        ChunkedOrArrayT,
        ChunkedOrScalar,
        ChunkedOrScalarAny,
        ChunkedOrScalarT,
        ChunkedStruct,
        DataType,
        DataTypeRemap,
        DataTypeT,
        DateScalar,
        IntegerScalar,
        IntegerType,
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
        ScalarT,
        StringScalar,
        StringType,
        StructArray,
        UInt32Type,
        UnaryFunction,
        UnaryNumeric,
        VectorFunction,
    )
    from narwhals._plan.compliant.typing import SeriesT
    from narwhals._plan.options import RankOptions, SortMultipleOptions
    from narwhals._plan.typing import Seq
    from narwhals._typing import NoDefault
    from narwhals.typing import (
        ClosedInterval,
        FillNullStrategy,
        IntoArrowSchema,
        IntoDType,
        NonNestedLiteral,
        NumericLiteral,
        PythonLiteral,
        UniqueKeepStrategy,
    )

    Ts = TypeVarTuple("Ts")

# NOTE: Common data type instances to share
UI32: Final = pa.uint32()
I64: Final = pa.int64()
F64: Final = pa.float64()
BOOL: Final = pa.bool_()

EMPTY: Final = ""
"""The empty string."""


class MinMax(ir.AggExpr):
    """Returns a `Struct({'min': ..., 'max': ...})`.

    https://arrow.apache.org/docs/python/generated/pyarrow.compute.min_max.html#pyarrow.compute.min_max
    """


IntoColumnAgg: TypeAlias = Callable[[str], ir.AggExpr]
"""Helper constructor for single-column aggregations."""

is_null = t.cast("UnaryFunction[ScalarAny, pa.BooleanScalar]", pc.is_null)
is_not_null = t.cast("UnaryFunction[ScalarAny, pa.BooleanScalar]", pc.is_valid)
is_nan = t.cast("UnaryFunction[ScalarAny, pa.BooleanScalar]", pc.is_nan)
is_finite = t.cast("UnaryFunction[ScalarAny, pa.BooleanScalar]", pc.is_finite)
not_ = t.cast("UnaryFunction[ScalarAny, pa.BooleanScalar]", pc.invert)


@overload
def is_not_nan(native: ChunkedArrayAny) -> ChunkedArray[pa.BooleanScalar]: ...
@overload
def is_not_nan(native: ScalarAny) -> pa.BooleanScalar: ...
@overload
def is_not_nan(native: ChunkedOrScalarAny) -> ChunkedOrScalar[pa.BooleanScalar]: ...
@overload
def is_not_nan(native: Arrow[ScalarAny]) -> Arrow[pa.BooleanScalar]: ...
def is_not_nan(native: Arrow[ScalarAny]) -> Arrow[pa.BooleanScalar]:
    return not_(is_nan(native))


and_ = t.cast("BinaryLogical", pc.and_kleene)
or_ = t.cast("BinaryLogical", pc.or_kleene)
xor = t.cast("BinaryLogical", pc.xor)

eq = t.cast("BinaryComp", pc.equal)
not_eq = t.cast("BinaryComp", pc.not_equal)
gt_eq = t.cast("BinaryComp", pc.greater_equal)
gt = t.cast("BinaryComp", pc.greater)
lt_eq = t.cast("BinaryComp", pc.less_equal)
lt = t.cast("BinaryComp", pc.less)


add = t.cast("BinaryNumericTemporal", pc.add)
sub = t.cast("BinaryNumericTemporal", pc.subtract)
multiply = pc.multiply
power = t.cast("BinaryFunction[NumericScalar, NumericScalar]", pc.power)
floordiv = _floordiv
abs_ = t.cast("UnaryNumeric", pc.abs)
exp = t.cast("UnaryNumeric", pc.exp)
sqrt = t.cast("UnaryNumeric", pc.sqrt)
ceil = t.cast("UnaryNumeric", pc.ceil)
floor = t.cast("UnaryNumeric", pc.floor)


def truediv(lhs: Incomplete, rhs: Incomplete) -> Incomplete:
    return pc.divide(*cast_for_truediv(lhs, rhs))


def modulus(lhs: Incomplete, rhs: Incomplete) -> Incomplete:
    floor_div = floordiv(lhs, rhs)
    return sub(lhs, multiply(floor_div, rhs))


# TODO @dangotbanned: Somehow fix the typing on this
# - `_ArrowDispatch` is relying on the gradual typing
_DISPATCH_BINARY: Mapping[type[ops.Operator], BinOp] = {
    # BinaryComp
    ops.Eq: eq,
    ops.NotEq: not_eq,
    ops.Lt: lt,
    ops.LtEq: lt_eq,
    ops.Gt: gt,
    ops.GtEq: gt_eq,
    # BinaryFunction (well it should be)
    ops.Add: add,  # BinaryNumericTemporal
    ops.Sub: sub,  # pyarrow-stubs
    ops.Multiply: multiply,  # pyarrow-stubs
    ops.TrueDivide: truediv,  # [[Any, Any], Any]
    ops.FloorDivide: floordiv,  # [[ArrayOrScalar, ArrayOrScalar], Any]
    ops.Modulus: modulus,  # [[Any, Any], Any]
    # BinaryLogical
    ops.And: and_,
    ops.Or: or_,
    ops.ExclusiveOr: xor,
}


def bin_op(
    function: Callable[[Any, Any], Any], /, *, reflect: bool = False
) -> Callable[[SeriesT, Any], SeriesT]:
    """Attach a binary operator to `ArrowSeries`."""

    def f(self: SeriesT, other: SeriesT | Any, /) -> SeriesT:
        right = other.native if isinstance(other, type(self)) else lit(other)
        return self._with_native(function(self.native, right))

    def f_reflect(self: SeriesT, other: SeriesT | Any, /) -> SeriesT:
        if isinstance(other, type(self)):
            name = other.name
            right: ArrowAny = other.native
        else:
            name = "literal"
            right = lit(other)
        return self.from_native(function(right, self.native), name, version=self.version)

    return f_reflect if reflect else f


_IS_BETWEEN: Mapping[ClosedInterval, tuple[BinaryComp, BinaryComp]] = {
    "left": (gt_eq, lt),
    "right": (gt, lt_eq),
    "none": (gt, lt),
    "both": (gt_eq, lt_eq),
}


@t.overload
def dtype_native(dtype: IntoDType, version: Version) -> pa.DataType: ...
@t.overload
def dtype_native(dtype: None, version: Version) -> None: ...
@t.overload
def dtype_native(dtype: IntoDType | None, version: Version) -> pa.DataType | None: ...
def dtype_native(dtype: IntoDType | None, version: Version) -> pa.DataType | None:
    return dtype if dtype is None else _dtype_native(dtype, version)


@t.overload
def cast(
    native: Scalar[Any], target_type: DataTypeT, *, safe: bool | None = ...
) -> Scalar[DataTypeT]: ...
@t.overload
def cast(
    native: ChunkedArray[Any], target_type: DataTypeT, *, safe: bool | None = ...
) -> ChunkedArray[Scalar[DataTypeT]]: ...
@t.overload
def cast(
    native: ChunkedOrScalar[Scalar[Any]],
    target_type: DataTypeT,
    *,
    safe: bool | None = ...,
) -> ChunkedArray[Scalar[DataTypeT]] | Scalar[DataTypeT]: ...
def cast(
    native: ChunkedOrScalar[Scalar[Any]],
    target_type: DataTypeT,
    *,
    safe: bool | None = None,
) -> ChunkedArray[Scalar[DataTypeT]] | Scalar[DataTypeT]:
    return pc.cast(native, target_type, safe=safe)


def cast_schema(
    native: pa.Schema, target_types: DataType | Mapping[str, DataType] | DataTypeRemap
) -> pa.Schema:
    if isinstance(target_types, pa.DataType):
        return pa.schema((name, target_types) for name in native.names)
    if _is_into_pyarrow_schema(target_types):
        new_schema = native
        for name, dtype in target_types.items():
            index = native.get_field_index(name)
            new_schema.set(index, native.field(index).with_type(dtype))
        return new_schema
    return pa.schema((fld.name, target_types.get(fld.type, fld.type)) for fld in native)


def cast_table(
    native: pa.Table, target: DataType | IntoArrowSchema | DataTypeRemap
) -> pa.Table:
    s = target if isinstance(target, pa.Schema) else cast_schema(native.schema, target)
    return native.cast(s)


def has_large_string(data_types: Iterable[DataType], /) -> bool:
    return any(pa.types.is_large_string(tp) for tp in data_types)


def string_type(data_types: Iterable[DataType] = (), /) -> StringType:
    """Return a native string type, compatible with `data_types`.

    Until [apache/arrow#45717] is resolved, we need to upcast `string` to `large_string` when joining.

    [apache/arrow#45717]: https://github.com/apache/arrow/issues/45717
    """
    return pa.large_string() if has_large_string(data_types) else pa.string()


# NOTE: `mypy` isn't happy, but this broadcasting behavior is worth documenting
@t.overload
def struct(names: Iterable[str], columns: Iterable[ChunkedArrayAny]) -> ChunkedStruct: ...
@t.overload
def struct(names: Iterable[str], columns: Iterable[ArrayAny]) -> pa.StructArray: ...
@t.overload
def struct(  # type: ignore[overload-overlap]
    names: Iterable[str], columns: Iterable[ScalarAny] | Iterable[NonNestedLiteral]
) -> pa.StructScalar: ...
@t.overload
def struct(  # type: ignore[overload-overlap]
    names: Iterable[str], columns: Iterable[ChunkedArrayAny | NonNestedLiteral]
) -> ChunkedStruct: ...
@t.overload
def struct(
    names: Iterable[str], columns: Iterable[ArrayAny | NonNestedLiteral]
) -> pa.StructArray: ...
@t.overload
def struct(names: Iterable[str], columns: Iterable[ArrowAny]) -> Incomplete: ...
def struct(names: Iterable[str], columns: Iterable[Incomplete]) -> Incomplete:
    """Collect columns into a struct.

    Arguments:
        names: Names of the struct fields to create.
        columns: Value(s) to collect into a struct. Scalars will will be broadcast unless all
            inputs are scalar.
    """
    return pc.make_struct(
        *columns, options=pc.MakeStructOptions(common.ensure_seq_str(names))
    )


def struct_schema(native: Arrow[pa.StructScalar] | pa.StructType) -> pa.Schema:
    """Get the struct definition as a schema."""
    tp = native.type if _is_arrow(native) else native
    fields = tp.fields if compat.HAS_STRUCT_TYPE_FIELDS else list(tp)
    return pa.schema(fields)


def struct_field_names(native: Arrow[pa.StructScalar] | pa.StructType) -> list[str]:
    """Get the names of all struct fields."""
    tp = native.type if _is_arrow(native) else native
    return tp.names if compat.HAS_STRUCT_TYPE_FIELDS else [f.name for f in tp]


@t.overload
def struct_field(native: ChunkedStruct, field: Field, /) -> ChunkedArrayAny: ...
@t.overload
def struct_field(native: StructArray, field: Field, /) -> ArrayAny: ...
@t.overload
def struct_field(native: pa.StructScalar, field: Field, /) -> ScalarAny: ...
@t.overload
def struct_field(native: SameArrowT, field: Field, /) -> SameArrowT: ...
@t.overload
def struct_field(native: ChunkedOrScalarAny, field: Field, /) -> ChunkedOrScalarAny: ...
def struct_field(native: ArrowAny, field: Field, /) -> ArrowAny:
    """Retrieve one `Struct` field."""
    func = t.cast("Callable[[Any,Any], ArrowAny]", pc.struct_field)
    return func(native, field)


@t.overload
def struct_fields(native: ChunkedStruct, *fields: Field) -> Seq[ChunkedArrayAny]: ...
@t.overload
def struct_fields(native: StructArray, *fields: Field) -> Seq[ArrayAny]: ...
@t.overload
def struct_fields(native: pa.StructScalar, *fields: Field) -> Seq[ScalarAny]: ...
@t.overload
def struct_fields(native: SameArrowT, *fields: Field) -> Seq[SameArrowT]: ...
def struct_fields(native: ArrowAny, *fields: Field) -> Seq[ArrowAny]:
    """Retrieve  multiple `Struct` fields."""
    func = t.cast("Callable[[Any,Any], ArrowAny]", pc.struct_field)
    return tuple(func(native, name) for name in fields)


def get_categories(native: ArrowAny) -> ChunkedArrayAny:
    da: Incomplete
    if isinstance(native, pa.ChunkedArray):
        da = native.unify_dictionaries().chunk(0)
    else:
        da = native
    return chunked_array(da.dictionary)


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


def str_join(
    native: Arrow[StringScalar], separator: str, *, ignore_nulls: bool = True
) -> StringScalar:
    """Vertically concatenate the string values in the column to a single string value."""
    if isinstance(native, pa.Scalar):
        # already joined
        return native
    if ignore_nulls and native.null_count:
        native = native.drop_null()
    return list_join_scalar(implode(native), separator, ignore_nulls=False)


def str_len_chars(native: ChunkedOrScalarAny) -> ChunkedOrScalarAny:
    len_chars: Incomplete = pc.utf8_length
    result: ChunkedOrScalarAny = len_chars(native)
    return result


def str_slice(
    native: ChunkedOrScalarAny, offset: int, length: int | None = None
) -> ChunkedOrScalarAny:
    stop = length if length is None else offset + length
    return pc.utf8_slice_codeunits(native, offset, stop=stop)


def str_pad_start(
    native: ChunkedOrScalarAny, length: int, fill_char: str = " "
) -> ChunkedOrScalarAny:  # pragma: no cover
    return pc.utf8_lpad(native, length, fill_char)


@t.overload
def str_find(
    native: ChunkedArrayAny,
    pattern: str,
    *,
    literal: bool = ...,
    not_found: int | None = ...,
) -> ChunkedArray[IntegerScalar]: ...
@t.overload
def str_find(
    native: Array, pattern: str, *, literal: bool = ..., not_found: int | None = ...
) -> Array[IntegerScalar]: ...
@t.overload
def str_find(
    native: ScalarAny, pattern: str, *, literal: bool = ..., not_found: int | None = ...
) -> IntegerScalar: ...
def str_find(
    native: Arrow[StringScalar],
    pattern: str,
    *,
    literal: bool = False,
    not_found: int | None = -1,
) -> Arrow[IntegerScalar]:
    """Return the bytes offset of the first substring matching a pattern.

    To match `pl.Expr.str.find` behavior, pass `not_found=None`.

    Note:
        `pyarrow` distinguishes null *inputs* with `None` and failed matches with `-1`.
    """
    # NOTE: `pyarrow-stubs` uses concrete types here
    fn_name = "find_substring" if literal else "find_substring_regex"
    result: Arrow[IntegerScalar] = pc.call_function(
        fn_name, [native], pa_options.match_substring(pattern)
    )
    if not_found == -1:
        return result
    return when_then(eq(result, lit(-1)), lit(not_found, result.type), result)


_StringFunction0: TypeAlias = "Callable[[ChunkedOrScalarAny], ChunkedOrScalarAny]"
_StringFunction1: TypeAlias = "Callable[[ChunkedOrScalarAny, str], ChunkedOrScalarAny]"
str_starts_with = t.cast("_StringFunction1", pc.starts_with)
str_ends_with = t.cast("_StringFunction1", pc.ends_with)
str_to_uppercase = t.cast("_StringFunction0", pc.utf8_upper)
str_to_lowercase = t.cast("_StringFunction0", pc.utf8_lower)
str_to_titlecase = t.cast("_StringFunction0", pc.utf8_title)


def _str_split(
    native: ArrowAny, by: str, n: int | None = None, *, literal: bool = True
) -> Arrow[ListScalar]:
    name = "split_pattern" if literal else "split_pattern_regex"
    result: Arrow[ListScalar] = pc.call_function(
        name, [native], pa_options.split_pattern(by, n)
    )
    return result


@t.overload
def str_split(
    native: ChunkedArrayAny, by: str, *, literal: bool = ...
) -> ChunkedArray[ListScalar]: ...
@t.overload
def str_split(
    native: ChunkedOrScalarAny, by: str, *, literal: bool = ...
) -> ChunkedOrScalar[ListScalar]: ...
@t.overload
def str_split(native: ArrayAny, by: str, *, literal: bool = ...) -> pa.ListArray[Any]: ...
@t.overload
def str_split(native: ArrowAny, by: str, *, literal: bool = ...) -> Arrow[ListScalar]: ...
def str_split(native: ArrowAny, by: str, *, literal: bool = True) -> Arrow[ListScalar]:
    return _str_split(native, by, literal=literal)


@t.overload
def str_splitn(
    native: ChunkedArrayAny,
    by: str,
    n: int,
    *,
    literal: bool = ...,
    as_struct: bool = ...,
) -> ChunkedArray[ListScalar]: ...
@t.overload
def str_splitn(
    native: ChunkedOrScalarAny,
    by: str,
    n: int,
    *,
    literal: bool = ...,
    as_struct: bool = ...,
) -> ChunkedOrScalar[ListScalar]: ...
@t.overload
def str_splitn(
    native: ArrayAny, by: str, n: int, *, literal: bool = ..., as_struct: bool = ...
) -> pa.ListArray[Any]: ...
@t.overload
def str_splitn(
    native: ArrowAny, by: str, n: int, *, literal: bool = ..., as_struct: bool = ...
) -> Arrow[ListScalar]: ...
def str_splitn(
    native: ArrowAny, by: str, n: int, *, literal: bool = True, as_struct: bool = False
) -> Arrow[ListScalar]:
    """Split the string by a substring, restricted to returning at most `n` items."""
    result = _str_split(native, by, n, literal=literal)
    if as_struct:
        # NOTE: `polars` would return a struct w/ field names (`'field_0`, ..., 'field_n-1`)
        msg = "TODO: `ArrowExpr.str.splitn`"
        raise NotImplementedError(msg)
    return result


@t.overload
def str_contains(
    native: ChunkedArrayAny, pattern: str, *, literal: bool = ...
) -> ChunkedArray[pa.BooleanScalar]: ...
@t.overload
def str_contains(
    native: ChunkedOrScalarAny, pattern: str, *, literal: bool = ...
) -> ChunkedOrScalar[pa.BooleanScalar]: ...
@t.overload
def str_contains(
    native: ArrowAny, pattern: str, *, literal: bool = ...
) -> Arrow[pa.BooleanScalar]: ...
def str_contains(
    native: ArrowAny, pattern: str, *, literal: bool = False
) -> Arrow[pa.BooleanScalar]:
    """Check if the string contains a substring that matches a pattern."""
    name = "match_substring" if literal else "match_substring_regex"
    result: Arrow[pa.BooleanScalar] = pc.call_function(
        name, [native], pa_options.match_substring(pattern)
    )
    return result


def str_strip_chars(native: Incomplete, characters: str | None) -> Incomplete:
    if characters:
        return pc.utf8_trim(native, characters)
    return pc.utf8_trim_whitespace(native)


def str_replace(
    native: Incomplete, pattern: str, value: str, *, literal: bool = False, n: int = 1
) -> Incomplete:
    fn = pc.replace_substring if literal else pc.replace_substring_regex
    return fn(native, pattern, replacement=value, max_replacements=n)


def str_replace_all(
    native: Incomplete, pattern: str, value: str, *, literal: bool = False
) -> Incomplete:
    return str_replace(native, pattern, value, literal=literal, n=-1)


def str_replace_vector(
    native: ChunkedArrayAny,
    pattern: str,
    replacements: ChunkedArrayAny,
    *,
    literal: bool = False,
    n: int | None = 1,
) -> ChunkedArrayAny:
    has_match = str_contains(native, pattern, literal=literal)
    if not any_(has_match).as_py():
        # fastpath, no work to do
        return native
    match, match_replacements = filter_arrays(has_match, native, replacements)
    if n is None or n == -1:
        list_split_by = str_split(match, pattern, literal=literal)
    else:
        list_split_by = str_splitn(match, pattern, n + 1, literal=literal)
    replaced = list_join(list_split_by, match_replacements, ignore_nulls=False)
    if all_(has_match, ignore_nulls=False).as_py():
        return chunked_array(replaced)
    return replace_with_mask(native, has_match, array(replaced))


def str_zfill(native: ChunkedOrScalarAny, length: int) -> ChunkedOrScalarAny:
    if compat.HAS_ZFILL:
        zfill: Incomplete = pc.utf8_zero_fill  # type: ignore[attr-defined]
        result: ChunkedOrScalarAny = zfill(native, length)
    else:
        result = _str_zfill_compat(native, length)
    return result


# TODO @dangotbanned: Finish tidying this up
def _str_zfill_compat(
    native: ChunkedOrScalarAny, length: int
) -> Incomplete:  # pragma: no cover
    dtype = string_type([native.type])
    hyphen, plus = lit("-", dtype), lit("+", dtype)

    padded_remaining = str_pad_start(str_slice(native, 1), length - 1, "0")
    padded_lt_length = str_pad_start(native, length, "0")

    binary_join: Incomplete = pc.binary_join_element_wise
    if isinstance(native, pa.Scalar):
        case_1: ArrowAny = hyphen  # starts with hyphen and less than length
        case_2: ArrowAny = plus  # starts with plus and less than length
    else:
        arr_len = len(native)
        case_1 = repeat_unchecked(hyphen, arr_len)
        case_2 = repeat_unchecked(plus, arr_len)

    first_char = str_slice(native, 0, 1)
    lt_length = lt(str_len_chars(native), lit(length))
    first_hyphen_lt_length = and_(eq(first_char, hyphen), lt_length)
    first_plus_lt_length = and_(eq(first_char, plus), lt_length)
    return when_then(
        first_hyphen_lt_length,
        binary_join(case_1, padded_remaining, ""),
        when_then(
            first_plus_lt_length,
            binary_join(case_2, padded_remaining, ""),
            when_then(lt_length, padded_lt_length, native),
        ),
    )


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


def any_(native: Incomplete, *, ignore_nulls: bool = True) -> pa.BooleanScalar:
    return pc.any(native, min_count=0, skip_nulls=ignore_nulls)


def all_(native: Incomplete, *, ignore_nulls: bool = True) -> pa.BooleanScalar:
    return pc.all(native, min_count=0, skip_nulls=ignore_nulls)


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


def reverse(native: ChunkedOrArrayT) -> ChunkedOrArrayT:
    """Unlike other slicing ops, `[::-1]` creates a full-copy.

    https://github.com/apache/arrow/issues/19103#issuecomment-1377671886
    """
    return native[::-1]


def cum_sum(native: ChunkedOrArrayT) -> ChunkedOrArrayT:
    return pc.cumulative_sum(native, skip_nulls=True)


def cum_min(native: ChunkedOrArrayT) -> ChunkedOrArrayT:
    return pc.cumulative_min(native, skip_nulls=True)


def cum_max(native: ChunkedOrArrayT) -> ChunkedOrArrayT:
    return pc.cumulative_max(native, skip_nulls=True)


def cum_prod(native: ChunkedOrArrayT) -> ChunkedOrArrayT:
    return pc.cumulative_prod(native, skip_nulls=True)


def cum_count(native: ChunkedArrayAny) -> ChunkedArrayAny:
    return cum_sum(is_not_null(native).cast(pa.uint32()))


_CUMULATIVE: Mapping[type[F.CumAgg], Callable[[ChunkedArrayAny], ChunkedArrayAny]] = {
    F.CumSum: cum_sum,
    F.CumCount: cum_count,
    F.CumMin: cum_min,
    F.CumMax: cum_max,
    F.CumProd: cum_prod,
}


def cumulative(native: ChunkedArrayAny, f: F.CumAgg, /) -> ChunkedArrayAny:
    func = _CUMULATIVE[type(f)]
    return func(native) if not f.reverse else reverse(func(reverse(native)))


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


def is_only_nulls(native: ChunkedOrArrayAny, *, nan_is_null: bool = False) -> bool:
    """Return True if `native` has no non-null values (and optionally include NaN)."""
    return array(native.is_null(nan_is_null=nan_is_null), BOOL).false_count == 0


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


@t.overload
def is_between(
    native: ChunkedArray[ScalarT],
    lower: ChunkedOrScalar[ScalarT] | NumericLiteral,
    upper: ChunkedOrScalar[ScalarT] | NumericLiteral,
    closed: ClosedInterval,
) -> ChunkedArray[pa.BooleanScalar]: ...
@t.overload
def is_between(
    native: ChunkedOrScalar[ScalarT],
    lower: ChunkedOrScalar[ScalarT] | NumericLiteral,
    upper: ChunkedOrScalar[ScalarT] | NumericLiteral,
    closed: ClosedInterval,
) -> ChunkedOrScalar[pa.BooleanScalar]: ...
def is_between(
    native: ChunkedOrScalar[ScalarT],
    lower: ChunkedOrScalar[ScalarT] | NumericLiteral,
    upper: ChunkedOrScalar[ScalarT] | NumericLiteral,
    closed: ClosedInterval,
) -> ChunkedOrScalar[pa.BooleanScalar]:
    fn_lhs, fn_rhs = _IS_BETWEEN[closed]
    low, high = (el if _is_arrow(el) else lit(el) for el in (lower, upper))
    out: ChunkedOrScalar[pa.BooleanScalar] = and_(
        fn_lhs(native, low), fn_rhs(native, high)
    )
    return out


@t.overload
def is_in(
    values: ChunkedArrayAny, /, other: ChunkedOrArrayAny
) -> ChunkedArray[pa.BooleanScalar]: ...
@t.overload
def is_in(values: ArrayAny, /, other: ChunkedOrArrayAny) -> Array[pa.BooleanScalar]: ...
@t.overload
def is_in(values: ScalarAny, /, other: ChunkedOrArrayAny) -> pa.BooleanScalar: ...
@t.overload
def is_in(
    values: ChunkedOrScalarAny, /, other: ChunkedOrArrayAny
) -> ChunkedOrScalarAny: ...
def is_in(values: ArrowAny, /, other: ChunkedOrArrayAny) -> ArrowAny:
    """Check if elements of `values` are present in `other`.

    Roughly equivalent to [`polars.Expr.is_in`](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.is_in.html)

    Returns a mask with `len(values)` elements.
    """
    # NOTE: Stubs don't include a `ChunkedArray` return
    # NOTE: Replaced ambiguous parameter name (`value_set`)
    is_in_: Incomplete = pc.is_in
    return is_in_(values, other)  # type: ignore[no-any-return]


@t.overload
def eq_missing(
    native: ChunkedArrayAny, other: NonNestedLiteral | ArrowAny
) -> ChunkedArray[pa.BooleanScalar]: ...
@t.overload
def eq_missing(
    native: ArrayAny, other: NonNestedLiteral | ArrowAny
) -> Array[pa.BooleanScalar]: ...
@t.overload
def eq_missing(
    native: ScalarAny, other: NonNestedLiteral | ArrowAny
) -> pa.BooleanScalar: ...
@t.overload
def eq_missing(
    native: ChunkedOrScalarAny, other: NonNestedLiteral | ArrowAny
) -> ChunkedOrScalarAny: ...
def eq_missing(native: ArrowAny, other: NonNestedLiteral | ArrowAny) -> ArrowAny:
    """Equivalent to `native == other` where `None == None`.

    This differs from default `eq` where null values are propagated.

    Note:
        Unique to `pyarrow`, this wrapper will ensure `None` uses `native.type`.
    """
    if isinstance(other, (pa.Array, pa.ChunkedArray)):
        return is_in(native, other)
    item = array(other if isinstance(other, pa.Scalar) else lit(other, native.type))
    return is_in(native, item)


def ir_min_max(name: str, /) -> MinMax:
    return MinMax(expr=ir.col(name))


def _boolean_is_unique(
    indices: ChunkedArrayAny, aggregated: ChunkedStruct, /
) -> ChunkedArrayAny:
    min, max = struct_fields(aggregated, "min", "max")
    return and_(is_in(indices, min), is_in(indices, max))


def _boolean_is_duplicated(
    indices: ChunkedArrayAny, aggregated: ChunkedStruct, /
) -> ChunkedArrayAny:
    return not_(_boolean_is_unique(indices, aggregated))


BOOLEAN_LENGTH_PRESERVING: Mapping[
    type[ir.boolean.BooleanFunction], tuple[IntoColumnAgg, BooleanLengthPreserving]
] = {
    ir.boolean.IsFirstDistinct: (ir.min, is_in),
    ir.boolean.IsLastDistinct: (ir.max, is_in),
    ir.boolean.IsUnique: (ir_min_max, _boolean_is_unique),
    ir.boolean.IsDuplicated: (ir_min_max, _boolean_is_duplicated),
}
_UNIQUE_KEEP_BOOLEAN_LENGTH_PRESERVING: Mapping[
    UniqueKeepStrategy, type[ir.boolean.BooleanFunction]
] = {
    "any": ir.boolean.IsFirstDistinct,
    "first": ir.boolean.IsFirstDistinct,
    "last": ir.boolean.IsLastDistinct,
    "none": ir.boolean.IsUnique,
}


def unique_keep_boolean_length_preserving(
    keep: UniqueKeepStrategy,
) -> tuple[IntoColumnAgg, BooleanLengthPreserving]:
    return BOOLEAN_LENGTH_PRESERVING[_UNIQUE_KEEP_BOOLEAN_LENGTH_PRESERVING[keep]]


def binary(
    lhs: ChunkedOrScalarAny, op: type[ops.Operator], rhs: ChunkedOrScalarAny
) -> ChunkedOrScalarAny:
    return _DISPATCH_BINARY[op](lhs, rhs)


@t.overload
def concat_str(
    *arrays: ChunkedArrayAny, separator: str = ..., ignore_nulls: bool = ...
) -> ChunkedArray[StringScalar]: ...
@t.overload
def concat_str(
    *arrays: ArrayAny, separator: str = ..., ignore_nulls: bool = ...
) -> Array[StringScalar]: ...
@t.overload
def concat_str(
    *arrays: ScalarAny, separator: str = ..., ignore_nulls: bool = ...
) -> StringScalar: ...
def concat_str(
    *arrays: ArrowAny, separator: str = "", ignore_nulls: bool = False
) -> Arrow[StringScalar]:
    """Horizontally arrow data into a single string column."""
    dtype = string_type(obj.type for obj in arrays)
    it = (obj.cast(dtype) for obj in arrays)
    concat: Incomplete = pc.binary_join_element_wise
    join = pa_options.join(ignore_nulls=ignore_nulls)
    return concat(*it, lit(separator, dtype), options=join)  # type: ignore[no-any-return]


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


@overload
def int_range(
    start: int = ...,
    end: int | None = ...,
    step: int = ...,
    /,
    *,
    dtype: IntegerType = ...,
    chunked: Literal[True] = ...,
) -> ChunkedArray[IntegerScalar]: ...
@overload
def int_range(
    start: int = ...,
    end: int | None = ...,
    step: int = ...,
    /,
    *,
    chunked: Literal[False],
) -> pa.Int64Array: ...
@overload
def int_range(
    start: int = ...,
    end: int | None = ...,
    step: int = ...,
    /,
    *,
    dtype: IntegerType = ...,
    chunked: Literal[False],
) -> Array[IntegerScalar]: ...
def int_range(
    start: int = 0,
    end: int | None = None,
    step: int = 1,
    /,
    *,
    dtype: IntegerType = I64,
    chunked: bool = True,
) -> ChunkedOrArray[IntegerScalar]:
    if end is None:
        end = start
        start = 0
    if not compat.HAS_ARANGE:  # pragma: no cover
        import numpy as np  # ignore-banned-import

        arr = pa.array(np.arange(start, end, step), type=dtype)
    else:
        int_range_: Incomplete = pa.arange  # type: ignore[attr-defined]
        arr = t.cast("ArrayAny", int_range_(start, end, step)).cast(dtype)
    return arr if not chunked else pa.chunked_array([arr])


def date_range(
    start: dt.date,
    end: dt.date,
    interval: int,  # (* assuming the `Interval` part is solved)
    *,
    closed: ClosedInterval = "both",
) -> ChunkedArray[DateScalar]:
    start_i = pa.scalar(start).cast(pa.int32()).as_py()
    end_i = pa.scalar(end).cast(pa.int32()).as_py()
    ca = int_range(start_i, end_i + 1, interval, dtype=pa.int32())
    if closed == "both":
        return ca.cast(pa.date32())
    if closed == "left":
        ca = ca.slice(length=ca.length() - 1)
    elif closed == "none":
        ca = ca.slice(1, length=ca.length() - 1)
    else:
        ca = ca.slice(1)
    return ca.cast(pa.date32())


def linear_space(
    start: float, end: float, num_samples: int, *, closed: ClosedInterval = "both"
) -> ChunkedArray[pc.NumericScalar]:
    """Based on [`new_linear_space_f64`].

    [`new_linear_space_f64`]: https://github.com/pola-rs/polars/blob/1684cc09dfaa46656dfecc45ab866d01aa69bc78/crates/polars-ops/src/series/ops/linear_space.rs#L62-L94
    """
    if num_samples < 0:
        msg = f"Number of samples, {num_samples}, must be non-negative."
        raise ValueError(msg)
    if num_samples == 0:
        return chunked_array([[]], F64)
    if num_samples == 1:
        if closed == "none":
            value = (end + start) * 0.5
        elif closed in {"left", "both"}:
            value = float(start)
        else:
            value = float(end)
        return chunked_array([[value]], F64)
    n = num_samples
    span = float(end - start)
    if closed == "none":
        d = span / (n + 1)
        start = start + d
    elif closed == "left":
        d = span / n
    elif closed == "right":
        start = start + span / n
        d = span / n
    else:
        d = span / (n - 1)
    ca: ChunkedArray[pc.NumericScalar] = multiply(int_range(0, n).cast(F64), lit(d))
    ca = add(ca, lit(start, F64))
    return ca  # noqa: RET504


def repeat(value: ScalarAny | NonNestedLiteral, n: int) -> ArrayAny:
    value = value if isinstance(value, pa.Scalar) else lit(value)
    return repeat_unchecked(value, n)


def repeat_unchecked(value: ScalarAny, /, n: int) -> ArrayAny:
    repeat_: Incomplete = pa.repeat
    result: ArrayAny = repeat_(value, n)
    return result


def repeat_like(value: NonNestedLiteral, n: int, native: ArrowAny) -> ArrayAny:
    return repeat_unchecked(lit(value, native.type), n)


def nulls_like(n: int, native: ArrowAny) -> ArrayAny:
    """Create a strongly-typed Array instance with all elements null.

    Uses the type of `native`.
    """
    result: ArrayAny = pa.nulls(n, native.type)
    return result


def zeros(n: int, /) -> pa.Int64Array:
    return pa.repeat(0, n)


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
    values, counts = struct_fields(value_counts, "values", "counts")
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


@overload
def lit(value: Any) -> NativeScalar: ...
@overload
def lit(value: Any, dtype: BoolType) -> pa.BooleanScalar: ...
@overload
def lit(value: Any, dtype: UInt32Type) -> pa.UInt32Scalar: ...
@overload
def lit(value: Any, dtype: DataType | None = ...) -> NativeScalar: ...
def lit(value: Any, dtype: DataType | None = None) -> NativeScalar:
    return pa.scalar(value) if dtype is None else pa.scalar(value, dtype)


# TODO @dangotbanned: Report `ListScalar.values` bug upstream
# See `tests/plan/list_unique_test.py::test_list_unique_scalar[None-None]`
@overload
def array(data: ArrowAny, /) -> ArrayAny: ...
@overload
def array(data: Arrow[BooleanScalar], dtype: BoolType, /) -> pa.BooleanArray: ...
@overload
def array(
    data: Iterable[PythonLiteral], dtype: DataType | None = None, /
) -> ArrayAny: ...
def array(
    data: ArrowAny | Iterable[PythonLiteral], dtype: DataType | None = None, /
) -> ArrayAny:
    """Convert `data` into an Array instance.

    Note:
        `dtype` is **not used** for existing `pyarrow` data, but it can be used to signal
        the concrete `Array` subclass that is returned.
        To actually changed the type, use `cast` instead.
    """
    if isinstance(data, pa.ChunkedArray):
        return data.combine_chunks()
    if isinstance(data, pa.Array):
        return data
    if isinstance(data, pa.Scalar):
        if isinstance(data, pa.ListScalar) and data.is_valid is False:
            return pa.array([None], data.type)
        return pa.array([data], data.type)
    return pa.array(data, dtype)


def chunked_array(
    data: ArrowAny | list[Iterable[Any]], dtype: DataType | None = None, /
) -> ChunkedArrayAny:
    return _chunked_array(array(data) if isinstance(data, pa.Scalar) else data, dtype)


def concat_horizontal(
    arrays: Collection[ChunkedOrArrayAny], names: Collection[str]
) -> pa.Table:
    """Concatenate `arrays` as columns in a new table."""
    table: Incomplete = pa.Table.from_arrays
    result: pa.Table = table(arrays, names)
    return result


def concat_vertical(
    arrays: Iterable[ChunkedOrArrayAny], dtype: DataType | None = None, /
) -> ChunkedArrayAny:
    """Concatenate `arrays` into a new array."""
    v_concat: Incomplete = pa.chunked_array
    result: ChunkedArrayAny = v_concat(arrays, dtype)
    return result


def to_table(array: ChunkedOrArrayAny, name: str = "") -> pa.Table:
    """Equivalent to `Series.to_frame`, but with an option to insert a name for the column."""
    return concat_horizontal((array,), (name,))


def _is_into_pyarrow_schema(obj: Mapping[Any, Any]) -> TypeIs[Mapping[str, DataType]]:
    return (
        (first := next(iter(obj.items())), None)
        and isinstance(first[0], str)
        and isinstance(first[1], pa.DataType)
    )


def _is_arrow(obj: Arrow[ScalarT] | Any) -> TypeIs[Arrow[ScalarT]]:
    return isinstance(obj, (pa.Scalar, pa.Array, pa.ChunkedArray))


def filter_arrays(
    predicate: ChunkedOrArray[BooleanScalar] | pc.Expression,
    *arrays: Unpack[Ts],
    ignore_nulls: bool = True,
) -> tuple[Unpack[Ts]]:
    """Apply the same filter to multiple arrays, returning them independently.

    Note:
        The typing here is a minefield. You'll get an `*arrays`-length `tuple[ChunkedArray, ...]`.
    """
    table: Incomplete = pa.Table.from_arrays
    tmp = [str(i) for i in range(len(arrays))]
    result = table(arrays, tmp).filter(predicate, "drop" if ignore_nulls else "emit_null")
    return t.cast("tuple[Unpack[Ts]]", tuple(result.columns))
