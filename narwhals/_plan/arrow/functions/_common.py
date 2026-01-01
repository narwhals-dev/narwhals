"""Things that still need a home."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pyarrow as pa  # ignore-banned-import

from narwhals._plan import expressions as ir

if TYPE_CHECKING:
    from typing_extensions import TypeIs

    from narwhals._plan.arrow.typing import Arrow, ChunkedOrArrayT, ScalarT


def is_arrow(obj: Arrow[ScalarT] | Any) -> TypeIs[Arrow[ScalarT]]:
    return isinstance(obj, (pa.Scalar, pa.Array, pa.ChunkedArray))


def reverse(native: ChunkedOrArrayT) -> ChunkedOrArrayT:
    """Return the array in reverse order.

    Important:
        Unlike other slicing operations, this [triggers a full-copy].

    [triggers a full-copy]: https://github.com/apache/arrow/issues/19103#issuecomment-1377671886
    """
    return native[::-1]


class MinMax(ir.AggExpr):
    """Returns a `Struct({'min': ..., 'max': ...})`.

    https://arrow.apache.org/docs/python/generated/pyarrow.compute.min_max.html#pyarrow.compute.min_max
    """
