from __future__ import annotations

from typing import TYPE_CHECKING, overload

import pyarrow as pa  # ignore-banned-import
import pyarrow.compute as pc  # ignore-banned-import

from narwhals._plan.arrow import compat, options as pa_options
from narwhals._plan.arrow.functions._arithmetic import multiply
from narwhals._plan.arrow.functions._construction import array, lit
from narwhals._plan.arrow.functions._dtypes import I64
from narwhals._plan.arrow.functions._ranges import int_range
from narwhals._plan.arrow.functions._round import round

if TYPE_CHECKING:
    from collections.abc import Sequence

    from typing_extensions import Unpack

    from narwhals._plan.arrow.typing import ArrayAny, ChunkedOrArrayAny, ChunkedOrArrayT
    from narwhals._plan.options import SortMultipleOptions, SortOptions

# TODO @dangotbanned: Module description

__all__ = ["random_indices", "reverse", "sort_indices", "unsort_indices"]


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
        pc.inverse_permutation(indices.cast(I64))  # type: ignore[attr-defined]
        if compat.HAS_SCATTER
        else int_range(len(indices), chunked=False).take(pc.sort_indices(indices))
    )


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


def reverse(native: ChunkedOrArrayT) -> ChunkedOrArrayT:
    """Return the array in reverse order.

    Important:
        Unlike other slicing operations, this [triggers a full-copy].

    [triggers a full-copy]: https://github.com/apache/arrow/issues/19103#issuecomment-1377671886
    """
    return native[::-1]
