"""Things that still need a home."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, overload

import pyarrow as pa  # ignore-banned-import
import pyarrow.compute as pc  # ignore-banned-import

from narwhals._plan import expressions as ir

if TYPE_CHECKING:
    from typing_extensions import TypeIs

    from narwhals._plan.arrow.typing import (
        Arrow,
        ArrowAny,
        ChunkedOrArrayT,
        ChunkedOrScalarAny,
        ScalarT,
    )


def is_arrow(obj: Arrow[ScalarT] | Any) -> TypeIs[Arrow[ScalarT]]:
    return isinstance(obj, (pa.Scalar, pa.Array, pa.ChunkedArray))


def reverse(native: ChunkedOrArrayT) -> ChunkedOrArrayT:
    """Return the array in reverse order.

    Important:
        Unlike other slicing operations, this [triggers a full-copy].

    [triggers a full-copy]: https://github.com/apache/arrow/issues/19103#issuecomment-1377671886
    """
    return native[::-1]


@overload
def round(native: ChunkedOrScalarAny, decimals: int = ...) -> ChunkedOrScalarAny: ...
@overload
def round(native: ChunkedOrArrayT, decimals: int = ...) -> ChunkedOrArrayT: ...
def round(native: ArrowAny, decimals: int = 0) -> ArrowAny:
    """Round underlying floating point data by `decimals` digits."""
    return pc.round(native, decimals, round_mode="half_towards_infinity")


class MinMax(ir.AggExpr):
    """Returns a `Struct({'min': ..., 'max': ...})`.

    https://arrow.apache.org/docs/python/generated/pyarrow.compute.min_max.html#pyarrow.compute.min_max
    """
