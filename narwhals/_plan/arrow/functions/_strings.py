"""String namespace functions."""

from __future__ import annotations

import typing as t
from typing import TYPE_CHECKING, Any, overload

import pyarrow as pa  # ignore-banned-import
import pyarrow.compute as pc  # ignore-banned-import

from narwhals._plan.arrow import compat, options as pa_options
from narwhals._plan.arrow.functions import _lists as list_
from narwhals._plan.arrow.functions._aggregation import implode
from narwhals._plan.arrow.functions._bin_op import and_, eq, lt
from narwhals._plan.arrow.functions._boolean import all_, any_
from narwhals._plan.arrow.functions._construction import (
    array,
    chunked_array,
    concat_horizontal,
    lit,
)
from narwhals._plan.arrow.functions._dtypes import string_type
from narwhals._plan.arrow.functions._multiplex import replace_with_mask, when_then
from narwhals._plan.arrow.functions._repeat import repeat_unchecked

if TYPE_CHECKING:
    from collections.abc import Callable

    from typing_extensions import TypeAlias

    from narwhals._arrow.typing import Incomplete
    from narwhals._plan.arrow.typing import (
        Array,
        ArrayAny,
        Arrow,
        ArrowAny,
        ChunkedArray,
        ChunkedArrayAny,
        ChunkedOrScalar,
        ChunkedOrScalarAny,
        IntegerScalar,
        ListScalar,
        ScalarAny,
        StringScalar,
    )

    _StringFunction0: TypeAlias = "Callable[[ChunkedOrScalarAny], ChunkedOrScalarAny]"
    _StringFunction1: TypeAlias = (
        "Callable[[ChunkedOrScalarAny, str], ChunkedOrScalarAny]"
    )


__all__ = [
    "concat_str",
    "contains",
    "ends_with",
    "find",
    "join",
    "len_chars",
    "pad_start",
    "replace",
    "replace_all",
    "replace_vector",
    "slice",
    "split",
    "splitn",
    "starts_with",
    "strip_chars",
    "to_lowercase",
    "to_titlecase",
    "to_uppercase",
    "zfill",
]

starts_with = t.cast("_StringFunction1", pc.starts_with)
"""Check if string values start with a substring."""

ends_with = t.cast("_StringFunction1", pc.ends_with)
"""Check if string values end with a substring."""

to_uppercase = t.cast("_StringFunction0", pc.utf8_upper)
"""Modify strings to their uppercase equivalent."""

to_lowercase = t.cast("_StringFunction0", pc.utf8_lower)
"""Modify strings to their lowercase equivalent."""

to_titlecase = t.cast("_StringFunction0", pc.utf8_title)
"""Modify strings to their titlecase equivalent."""


@overload
def concat_str(
    *arrays: ChunkedArrayAny, separator: str = ..., ignore_nulls: bool = ...
) -> ChunkedArray[StringScalar]: ...
@overload
def concat_str(
    *arrays: ArrayAny, separator: str = ..., ignore_nulls: bool = ...
) -> Array[StringScalar]: ...
@overload
def concat_str(
    *arrays: ScalarAny, separator: str = ..., ignore_nulls: bool = ...
) -> StringScalar: ...
def concat_str(
    *arrays: ArrowAny, separator: str = "", ignore_nulls: bool = False
) -> Arrow[StringScalar]:
    """Horizontally concatenate arrow data into a single string column."""
    dtype = string_type(obj.type for obj in arrays)
    it = (obj.cast(dtype) for obj in arrays)
    concat: Incomplete = pc.binary_join_element_wise
    join = pa_options.join(ignore_nulls=ignore_nulls)
    result: Arrow[StringScalar] = concat(*it, lit(separator, dtype), options=join)
    return result


def join(
    native: Arrow[StringScalar], separator: str, *, ignore_nulls: bool = True
) -> StringScalar:
    """Vertically concatenate the string values in the column to a single string value."""
    if isinstance(native, pa.Scalar):
        # already joined
        return native
    if ignore_nulls and native.null_count:
        native = native.drop_null()
    return list_.join_scalar(implode(native), separator, ignore_nulls=False)


def len_chars(native: ChunkedOrScalarAny) -> ChunkedOrScalarAny:
    """Return the length of each string as the number of characters."""
    len_chars: Incomplete = pc.utf8_length
    result: ChunkedOrScalarAny = len_chars(native)
    return result


def slice(
    native: ChunkedOrScalarAny, offset: int, length: int | None = None
) -> ChunkedOrScalarAny:
    """Extract a substring from each string value."""
    stop = length if length is None else offset + length
    return pc.utf8_slice_codeunits(native, offset, stop=stop)


def pad_start(
    native: ChunkedOrScalarAny, length: int, fill_char: str = " "
) -> ChunkedOrScalarAny:  # pragma: no cover
    """Pad the start of the string until it reaches the given length."""
    return pc.utf8_lpad(native, length, fill_char)


@overload
def find(
    native: ChunkedArrayAny,
    pattern: str,
    *,
    literal: bool = ...,
    not_found: int | None = ...,
) -> ChunkedArray[IntegerScalar]: ...
@overload
def find(
    native: Array, pattern: str, *, literal: bool = ..., not_found: int | None = ...
) -> Array[IntegerScalar]: ...
@overload
def find(
    native: ScalarAny, pattern: str, *, literal: bool = ..., not_found: int | None = ...
) -> IntegerScalar: ...
def find(
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


def _split(
    native: ArrowAny, by: str, n: int | None = None, *, literal: bool = True
) -> Arrow[ListScalar]:
    name = "split_pattern" if literal else "split_pattern_regex"
    result: Arrow[ListScalar] = pc.call_function(
        name, [native], pa_options.split_pattern(by, n)
    )
    return result


@overload
def split(
    native: ChunkedArrayAny, by: str, *, literal: bool = ...
) -> ChunkedArray[ListScalar]: ...
@overload
def split(
    native: ChunkedOrScalarAny, by: str, *, literal: bool = ...
) -> ChunkedOrScalar[ListScalar]: ...
@overload
def split(native: ArrayAny, by: str, *, literal: bool = ...) -> pa.ListArray[Any]: ...
@overload
def split(native: ArrowAny, by: str, *, literal: bool = ...) -> Arrow[ListScalar]: ...
def split(native: ArrowAny, by: str, *, literal: bool = True) -> Arrow[ListScalar]:
    """Split the string by a substring."""
    return _split(native, by, literal=literal)


# TODO @dangotbanned: Support and default to `as_struct=True`
# `polars` would return a struct w/ field names (`'field_0', ..., 'field_n-1'`)
@overload
def splitn(
    native: ChunkedArrayAny,
    by: str,
    n: int,
    *,
    literal: bool = ...,
    as_struct: bool = ...,
) -> ChunkedArray[ListScalar]: ...
@overload
def splitn(
    native: ChunkedOrScalarAny,
    by: str,
    n: int,
    *,
    literal: bool = ...,
    as_struct: bool = ...,
) -> ChunkedOrScalar[ListScalar]: ...
@overload
def splitn(
    native: ArrayAny, by: str, n: int, *, literal: bool = ..., as_struct: bool = ...
) -> pa.ListArray[Any]: ...
@overload
def splitn(
    native: ArrowAny, by: str, n: int, *, literal: bool = ..., as_struct: bool = ...
) -> Arrow[ListScalar]: ...
def splitn(
    native: ArrowAny, by: str, n: int, *, literal: bool = True, as_struct: bool = False
) -> Arrow[ListScalar]:
    """Split the string by a substring, restricted to returning at most `n` items."""
    result = _split(native, by, n, literal=literal)
    if as_struct:
        msg = "TODO: `ArrowExpr.str.splitn`"
        raise NotImplementedError(msg)
    return result


@overload
def contains(
    native: ChunkedArrayAny, pattern: str, *, literal: bool = ...
) -> ChunkedArray[pa.BooleanScalar]: ...
@overload
def contains(
    native: ChunkedOrScalarAny, pattern: str, *, literal: bool = ...
) -> ChunkedOrScalar[pa.BooleanScalar]: ...
@overload
def contains(
    native: ArrowAny, pattern: str, *, literal: bool = ...
) -> Arrow[pa.BooleanScalar]: ...
def contains(
    native: ArrowAny, pattern: str, *, literal: bool = False
) -> Arrow[pa.BooleanScalar]:
    """Check if the string contains a substring that matches a pattern."""
    name = "match_substring" if literal else "match_substring_regex"
    result: Arrow[pa.BooleanScalar] = pc.call_function(
        name, [native], pa_options.match_substring(pattern)
    )
    return result


def strip_chars(native: Incomplete, characters: str | None) -> Incomplete:
    """Remove leading and trailing characters."""
    if characters:
        return pc.utf8_trim(native, characters)
    return pc.utf8_trim_whitespace(native)


def replace(
    native: Incomplete, pattern: str, value: str, *, literal: bool = False, n: int = 1
) -> Incomplete:
    """Replace the first matching regex/literal substring with a new string value."""
    fn = pc.replace_substring if literal else pc.replace_substring_regex
    return fn(native, pattern, replacement=value, max_replacements=n)


def replace_all(
    native: Incomplete, pattern: str, value: str, *, literal: bool = False
) -> Incomplete:
    """Replace all matching regex/literal substrings with a new string value."""
    return replace(native, pattern, value, literal=literal, n=-1)


def replace_vector(
    native: ChunkedArrayAny,
    pattern: str,
    replacements: ChunkedArrayAny,
    *,
    literal: bool = False,
    n: int | None = 1,
) -> ChunkedArrayAny:
    """Replace the first matching regex/literal substring with the adjacent string in `replacements`."""
    has_match = contains(native, pattern, literal=literal)
    if not any_(has_match).as_py():
        # fastpath, no work to do
        return native
    match, match_replacements = (
        concat_horizontal([native, replacements], ["0", "1"]).filter(has_match).columns
    )
    if n is None or n == -1:
        list_split_by = split(match, pattern, literal=literal)
    else:
        list_split_by = splitn(match, pattern, n + 1, literal=literal)
    replaced = list_.join(list_split_by, match_replacements, ignore_nulls=False)
    if all_(has_match, ignore_nulls=False).as_py():
        return chunked_array(replaced)
    return replace_with_mask(native, has_match, array(replaced))


def zfill(native: ChunkedOrScalarAny, length: int) -> ChunkedOrScalarAny:
    """Pad the start of the string with zeros until it reaches the given length."""
    if compat.HAS_ZFILL:
        zfill: Incomplete = pc.utf8_zero_fill  # type: ignore[attr-defined]
        result: ChunkedOrScalarAny = zfill(native, length)
    else:
        result = _zfill_compat(native, length)
    return result


# TODO @dangotbanned: Finish tidying this up
def _zfill_compat(
    native: ChunkedOrScalarAny, length: int
) -> Incomplete:  # pragma: no cover
    dtype = string_type([native.type])
    hyphen, plus = lit("-", dtype), lit("+", dtype)

    padded_remaining = pad_start(slice(native, 1), length - 1, "0")
    padded_lt_length = pad_start(native, length, "0")

    binary_join: Incomplete = pc.binary_join_element_wise
    if isinstance(native, pa.Scalar):
        case_1: ArrowAny = hyphen  # starts with hyphen and less than length
        case_2: ArrowAny = plus  # starts with plus and less than length
    else:
        arr_len = len(native)
        case_1 = repeat_unchecked(hyphen, arr_len)
        case_2 = repeat_unchecked(plus, arr_len)

    first_char = slice(native, 0, 1)
    lt_length = lt(len_chars(native), lit(length))
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
