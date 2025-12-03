"""Native functions, aliased and/or with behavior aligned to `polars`."""

from __future__ import annotations

import math
import typing as t
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Final, Literal, overload

import pyarrow as pa  # ignore-banned-import
import pyarrow.compute as pc  # ignore-banned-import

from narwhals._arrow.utils import (
    cast_for_truediv,
    chunked_array as _chunked_array,
    floordiv_compat as _floordiv,
    narwhals_to_native_dtype as _dtype_native,
)
from narwhals._plan import expressions as ir
from narwhals._plan._guards import is_non_nested_literal
from narwhals._plan.arrow import options as pa_options
from narwhals._plan.expressions import functions as F, operators as ops
from narwhals._utils import Implementation, Version

if TYPE_CHECKING:
    import datetime as dt
    from collections.abc import Iterable, Mapping

    from typing_extensions import TypeAlias, TypeIs

    from narwhals._arrow.typing import Incomplete, PromoteOptions
    from narwhals._plan.arrow.acero import Field
    from narwhals._plan.arrow.typing import (
        Array,
        ArrayAny,
        Arrow,
        ArrowAny,
        ArrowT,
        BinaryComp,
        BinaryFunction,
        BinaryLogical,
        BinaryNumericTemporal,
        BinOp,
        BooleanLengthPreserving,
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
        ChunkedStruct,
        DataType,
        DataTypeRemap,
        DataTypeT,
        DateScalar,
        IntegerScalar,
        IntegerType,
        LargeStringType,
        ListArray,
        ListScalar,
        NativeScalar,
        NumericScalar,
        Predicate,
        SameArrowT,
        Scalar,
        ScalarAny,
        ScalarT,
        StringScalar,
        StringType,
        StructArray,
        UnaryFunction,
        UnaryNumeric,
        VectorFunction,
    )
    from narwhals._plan.compliant.typing import SeriesT
    from narwhals._plan.options import RankOptions, SortMultipleOptions, SortOptions
    from narwhals._plan.typing import Seq
    from narwhals.typing import (
        ClosedInterval,
        FillNullStrategy,
        IntoArrowSchema,
        IntoDType,
        NonNestedLiteral,
        PythonLiteral,
    )

BACKEND_VERSION = Implementation.PYARROW._backend_version()
"""Static backend version for `pyarrow`."""

RANK_ACCEPTS_CHUNKED: Final = BACKEND_VERSION >= (14,)

HAS_SCATTER: Final = BACKEND_VERSION >= (20,)
"""`pyarrow.compute.scatter` added in https://github.com/apache/arrow/pull/44394"""

HAS_KURTOSIS_SKEW = BACKEND_VERSION >= (20,)
"""`pyarrow.compute.{kurtosis,skew}` added in https://github.com/apache/arrow/pull/45677"""

HAS_ARANGE: Final = BACKEND_VERSION >= (21,)
"""`pyarrow.arange` added in https://github.com/apache/arrow/pull/46778"""

HAS_ZFILL: Final = BACKEND_VERSION >= (21,)
"""`pyarrow.compute.utf8_zero_fill` added in https://github.com/apache/arrow/pull/46815"""


I64: Final = pa.int64()
F64: Final = pa.float64()


class MinMax(ir.AggExpr):
    """Returns a `Struct({'min': ..., 'max': ...})`.

    https://arrow.apache.org/docs/python/generated/pyarrow.compute.min_max.html#pyarrow.compute.min_max
    """


IntoColumnAgg: TypeAlias = Callable[[str], ir.AggExpr]
"""Helper constructor for single-column aggregations."""

is_null = t.cast("UnaryFunction[ScalarAny, BooleanScalar]", pc.is_null)
is_not_null = t.cast("UnaryFunction[ScalarAny,BooleanScalar]", pc.is_valid)
is_nan = t.cast("UnaryFunction[ScalarAny, BooleanScalar]", pc.is_nan)
is_finite = t.cast("UnaryFunction[ScalarAny, BooleanScalar]", pc.is_finite)
not_ = t.cast("UnaryFunction[ScalarAny, BooleanScalar]", pc.invert)


@overload
def is_not_nan(native: ChunkedArrayAny) -> ChunkedArray[BooleanScalar]: ...
@overload
def is_not_nan(native: ScalarAny) -> BooleanScalar: ...
@overload
def is_not_nan(native: ChunkedOrScalarAny) -> ChunkedOrScalar[BooleanScalar]: ...
@overload
def is_not_nan(native: Arrow[ScalarAny]) -> Arrow[BooleanScalar]: ...
def is_not_nan(native: Arrow[ScalarAny]) -> Arrow[BooleanScalar]:
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


def string_type(data_types: Iterable[DataType] = (), /) -> StringType | LargeStringType:
    """Return a native string type, compatible with `data_types`.

    Until [apache/arrow#45717] is resolved, we need to upcast `string` to `large_string` when joining.

    [apache/arrow#45717]: https://github.com/apache/arrow/issues/45717
    """
    return pa.large_string() if has_large_string(data_types) else pa.string()


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


@t.overload
def list_len(native: ChunkedList) -> ChunkedArray[pa.UInt32Scalar]: ...
@t.overload
def list_len(native: ListArray) -> pa.UInt32Array: ...
@t.overload
def list_len(native: ListScalar) -> pa.UInt32Scalar: ...
@t.overload
def list_len(native: SameArrowT) -> SameArrowT: ...
@t.overload
def list_len(native: ChunkedOrScalar[ListScalar]) -> ChunkedOrScalar[pa.UInt32Scalar]: ...
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
    native: ListScalar[StringType],
    separator: Arrow[StringScalar] | str,
    *,
    ignore_nulls: bool = ...,
) -> pa.StringScalar: ...
def list_join(
    native: ArrowAny, separator: Arrow[StringScalar] | str, *, ignore_nulls: bool = False
) -> ArrowAny:
    """Join all string items in a sublist and place a separator between them.

    Each list of values in the first input is joined using each second input as separator.
    If any input list is null or contains a null, the corresponding output will be null.
    """
    if ignore_nulls:
        # NOTE: `polars` default is `True`, will need to handle that if this becomes api
        msg = "TODO: `ArrowExpr.list.join(ignore_nulls=True)`"
        raise NotImplementedError(msg)
    return pc.binary_join(native, separator)


def str_join(
    native: Arrow[StringScalar], separator: str, *, ignore_nulls: bool = True
) -> StringScalar:
    """Vertically concatenate the string values in the column to a single string value."""
    if isinstance(native, pa.Scalar):
        # already joined
        return native
    if ignore_nulls and native.null_count:
        native = native.drop_null()
    offsets = [0, len(native)]
    scalar = pa.ListArray.from_arrays(offsets, array(native))[0]
    return list_join(scalar, separator)


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
    tbl_matches = pa.Table.from_arrays([native, replacements], ["0", "1"]).filter(
        has_match
    )
    matches = tbl_matches.column(0)
    match_replacements = tbl_matches.column(1)
    if n is None or n == -1:
        list_split_by = str_split(matches, pattern, literal=literal)
    else:
        list_split_by = str_splitn(matches, pattern, n + 1, literal=literal)
    replaced = list_join(list_split_by, match_replacements)
    if all_(has_match, ignore_nulls=False).as_py():
        return chunked_array(replaced)
    return replace_with_mask(native, has_match, array(replaced))


def str_zfill(native: ChunkedOrScalarAny, length: int) -> ChunkedOrScalarAny:
    if HAS_ZFILL:
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
    if HAS_KURTOSIS_SKEW:
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
    arr = native if RANK_ACCEPTS_CHUNKED else array(native)
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


def replace_with_mask(
    native: ChunkedOrArrayT, mask: Predicate, replacements: ChunkedOrArrayAny
) -> ChunkedOrArrayT:
    if not isinstance(mask, pa.BooleanArray):
        mask = t.cast("pa.BooleanArray", array(mask))
    if not isinstance(replacements, pa.Array):
        replacements = array(replacements)
    result: ChunkedOrArrayT = pc.replace_with_mask(native, mask, replacements)
    return result


def is_between(
    native: ChunkedOrScalar[ScalarT],
    lower: ChunkedOrScalar[ScalarT],
    upper: ChunkedOrScalar[ScalarT],
    closed: ClosedInterval,
) -> ChunkedOrScalar[pa.BooleanScalar]:
    fn_lhs, fn_rhs = _IS_BETWEEN[closed]
    return and_(fn_lhs(native, lower), fn_rhs(native, upper))  # type: ignore[no-any-return]


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


def sort_indices(
    native: ChunkedOrArrayAny | pa.Table,
    *order_by: str,
    descending: bool | Sequence[bool] = False,
    nulls_last: bool = False,
    options: SortOptions | SortMultipleOptions | None = None,
) -> pa.UInt64Array:
    """Return the indices that would sort an array or table."""
    opts = (
        options.to_arrow(order_by)
        if options
        else pa_options.sort(*order_by, descending=descending, nulls_last=nulls_last)
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
        if HAS_SCATTER
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
    if not HAS_ARANGE:  # pragma: no cover
        import numpy as np  # ignore-banned-import

        arr = pa.array(np.arange(start=start, stop=end, step=step), type=dtype)
    else:
        int_range_: Incomplete = pa.arange  # type: ignore[attr-defined]
        arr = t.cast("ArrayAny", int_range_(start=start, stop=end, step=step)).cast(dtype)
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


def lit(value: Any, dtype: DataType | None = None) -> NativeScalar:
    return pa.scalar(value) if dtype is None else pa.scalar(value, dtype)


@overload
def array(data: ArrowAny, /) -> ArrayAny: ...
@overload
def array(
    data: Iterable[PythonLiteral], dtype: DataType | None = None, /
) -> ArrayAny: ...
def array(
    data: ArrowAny | Iterable[PythonLiteral], dtype: DataType | None = None, /
) -> ArrayAny:
    """Convert `data` into an Array instance.

    Note:
        `dtype` is not used for existing `pyarrow` data, use `cast` instead.
    """
    if isinstance(data, pa.ChunkedArray):
        return data.combine_chunks()
    if isinstance(data, pa.Array):
        return data
    if isinstance(data, pa.Scalar):
        return pa.array([data], data.type)
    return pa.array(data, dtype)


def chunked_array(
    data: ArrowAny | list[Iterable[Any]], dtype: DataType | None = None, /
) -> ChunkedArrayAny:
    return _chunked_array(array(data) if isinstance(data, pa.Scalar) else data, dtype)


def concat_vertical_chunked(
    arrays: Iterable[ChunkedOrArrayAny], dtype: DataType | None = None, /
) -> ChunkedArrayAny:
    v_concat: Incomplete = pa.chunked_array
    return v_concat(arrays, dtype)  # type: ignore[no-any-return]


def concat_vertical_table(
    tables: Iterable[pa.Table], /, promote_options: PromoteOptions = "none"
) -> pa.Table:
    return pa.concat_tables(tables, promote_options=promote_options)


if BACKEND_VERSION >= (14,):

    def concat_diagonal(tables: Iterable[pa.Table]) -> pa.Table:
        return pa.concat_tables(tables, promote_options="default")
else:

    def concat_diagonal(tables: Iterable[pa.Table]) -> pa.Table:
        return pa.concat_tables(tables, promote=True)


def _is_into_pyarrow_schema(obj: Mapping[Any, Any]) -> TypeIs[Mapping[str, DataType]]:
    return (
        (first := next(iter(obj.items())), None)
        and isinstance(first[0], str)
        and isinstance(first[1], pa.DataType)
    )
