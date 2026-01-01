"""Creating Arrow data and converting between representation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, overload

import pyarrow as pa  # ignore-banned-import

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable

    from typing_extensions import TypeAlias

    from narwhals._plan.arrow.typing import (
        ArrayAny,
        Arrow,
        ArrowAny,
        BooleanScalar,
        BoolType,
        ChunkedArrayAny,
        ChunkedOrArrayAny,
        DataType,
        ScalarAny,
        UInt32Type,
    )
    from narwhals.typing import NonNestedLiteral, PythonLiteral

Incomplete: TypeAlias = Any


@overload
def lit(value: Any) -> ScalarAny: ...
@overload
def lit(value: Any, dtype: BoolType) -> pa.BooleanScalar: ...
@overload
def lit(value: Any, dtype: UInt32Type) -> pa.UInt32Scalar: ...
@overload
def lit(value: Any, dtype: DataType | None = ...) -> ScalarAny: ...
def lit(value: Any, dtype: DataType | None = None) -> ScalarAny:
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
    arr = array(data) if isinstance(data, pa.Scalar) else data
    if isinstance(arr, pa.ChunkedArray):
        return arr
    arrs = [arr] if not isinstance(arr, list) else arr
    return pa.chunked_array(arrs) if dtype is None else pa.chunked_array(arrs, dtype)


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
