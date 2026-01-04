"""Categorical function namespace."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, overload

import pyarrow as pa  # ignore-banned-import

from narwhals._plan.arrow.functions._construction import array

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

    from narwhals._plan.arrow.typing import (
        ArrayAny,
        ArrowAny,
        ChunkedArrayAny,
        ChunkedOrArrayHashable,
    )

    Incomplete: TypeAlias = Any


__all__ = ["encode", "get_categories"]


def get_categories(native: ArrowAny, /) -> ChunkedArrayAny:
    """Get the categories stored in the data type.

    Arguments:
        native: Dictionary-typed arrow data.
    """
    da: Incomplete
    if isinstance(native, pa.ChunkedArray):
        da = native.unify_dictionaries().chunk(0)
    else:
        da = native
    return pa.chunked_array([da.dictionary])


@overload
def encode(native: ChunkedOrArrayHashable, /) -> pa.Int32Array: ...
@overload
def encode(
    native: ChunkedOrArrayHashable, /, *, include_categories: Literal[True]
) -> tuple[ArrayAny, pa.Int32Array]: ...
def encode(
    native: ChunkedOrArrayHashable, /, *, include_categories: bool = False
) -> tuple[ArrayAny, pa.Int32Array] | pa.Int32Array:
    """Return a [dictionary-encoded] version of the input array.

    Arguments:
        native: An arrow array.
        include_categories: Include the [`dictionary`] (categories) array in the output.

    Examples:
        >>> from narwhals._plan.arrow import functions as fn
        >>> values = [None, "foo", "bar", "foo", None, "foo", "ham"]
        >>> arr = fn.array(values)

        Return the underlying [`indices`] *into* the ordered categories

        >>> fn.cat.encode(arr).to_pylist()
        [0, 1, 2, 1, 0, 1, 3]

        Return the category each index refers to by specifying `include_categories=True`

        >>> categories, indices = fn.cat.encode(arr, include_categories=True)
        >>> categories.to_pylist(), indices.to_pylist()
        ([None, 'foo', 'bar', 'ham'], [0, 1, 2, 1, 0, 1, 3])

    [dictionary-encoded]: https://arrow.apache.org/docs/python/generated/pyarrow.compute.dictionary_encode.html#pyarrow.compute.dictionary_encode
    [`indices`]: https://arrow.apache.org/docs/python/generated/pyarrow.DictionaryArray.html#pyarrow.DictionaryArray.indices
    [`dictionary`]: https://arrow.apache.org/docs/python/generated/pyarrow.DictionaryArray.html#pyarrow.DictionaryArray.dictionary
    """
    da: Incomplete = array(native.dictionary_encode("encode"))
    indices: pa.Int32Array = da.indices
    if not include_categories:
        return indices
    categories: ArrayAny = da.dictionary
    return categories, indices
