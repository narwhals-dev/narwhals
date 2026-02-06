"""Creating Arrow data and converting between representations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, overload

import pyarrow as pa  # ignore-banned-import

from narwhals._arrow.utils import concat_tables

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
        IntoChunkedArray,
        ScalarAny,
        UInt32Type,
    )
    from narwhals.typing import PythonLiteral


__all__ = [
    "array",
    "chunked_array",
    "concat_horizontal",
    "concat_tables",
    "concat_vertical",
    "lit",
    "to_table",
]

Incomplete: TypeAlias = Any


@overload
def lit(value: Any, /) -> ScalarAny: ...
@overload
def lit(value: Any, /, dtype: BoolType) -> pa.BooleanScalar: ...
@overload
def lit(value: Any, /, dtype: UInt32Type) -> pa.UInt32Scalar: ...
@overload
def lit(value: Any, /, dtype: DataType | None = ...) -> ScalarAny: ...
def lit(value: Any, /, dtype: DataType | None = None) -> ScalarAny:
    """Convert `value` into a [`Scalar`].

    Note:
        Feel free to add more `@overload`s, but avoid matching on `value`'s type.
        If you need this, use [`pa.scalar`] directly but [pyarrow-stubs#208] may cause issues.

    [`Scalar`]: https://arrow.apache.org/docs/python/generated/pyarrow.Scalar.html
    [`pa.scalar`]: https://arrow.apache.org/docs/python/generated/pyarrow.scalar.html
    [pyarrow-stubs#208]: https://github.com/zen-xu/pyarrow-stubs/pull/208
    """
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
    """Convert `data` into an [`Array`].

    Note:
        `dtype` is **not used** for existing `pyarrow` data, but it can be used to signal
        the concrete `Array` subclass that is returned.
        To actually changed the type, use `cast` instead.

    [`Array`]: https://arrow.apache.org/docs/python/generated/pyarrow.Array.html
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
    data: IntoChunkedArray, dtype: DataType | None = None, /
) -> ChunkedArrayAny:
    """Convert `data` into a [`ChunkedArray`].

    Arguments:
        data: Anything than can be coerced into an array.
            A *little* more forgiving than [`pa.chunked_array`].
        dtype: A native `DataType`.

    Examples:
        The result of `lit` and `array` can be passed in directly for the same result

        >>> import pyarrow as pa
        >>> from narwhals._plan.arrow import functions as fn
        >>> one = fn.lit(1)
        >>> ones = fn.array(one)
        >>> fn.chunked_array(one).equals(fn.chunked_array(ones))
        True
        >>> fn.chunked_array(ones)  # doctest: +ELLIPSIS
        <pyarrow.lib.ChunkedArray object at ...>
        [
          [
            1
          ]
        ]

        An empty list and a `DataType` produce an empty array

        >>> fn.chunked_array([], pa.string())  # doctest: +ELLIPSIS
        <pyarrow.lib.ChunkedArray object at ...>
        [

        ]

        Chunks can be specified using nested lists

        >>> short = fn.chunked_array([[1], [2, 2], [3]])
        >>> [c.to_pylist() for c in short.chunks]
        [[1], [2, 2], [3]]

        Which is equivalent to using `array` for each chunk

        >>> longer = fn.chunked_array([fn.array([1]), fn.array([2, 2]), fn.array([3])])
        >>> short.equals(longer)
        True

        If you're feeling funky, all of these guys work as well

        >>> import numpy as np
        >>> import pandas as pd
        >>> import polars as pl
        >>> im_surprised_too = [
        ...     pl.Series([1]),
        ...     np.array([2]),
        ...     pd.Series([3]),
        ...     fn.array([4]),
        ... ]
        >>> [c.to_pylist() for c in fn.chunked_array(im_surprised_too).chunks]
        [[1], [2], [3], [4]]

    [`ChunkedArray`]: https://arrow.apache.org/docs/python/generated/pyarrow.ChunkedArray.html
    [`pa.chunked_array`]: https://arrow.apache.org/docs/python/generated/pyarrow.chunked_array.html#pyarrow.chunked_array
    """
    arr = array(data) if isinstance(data, pa.Scalar) else data
    if isinstance(arr, pa.ChunkedArray):
        return arr
    func: Incomplete = pa.chunked_array
    result: ChunkedArrayAny = func([arr] if not isinstance(arr, list) else arr, dtype)
    return result


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
