"""Things that still need a home."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pyarrow as pa  # ignore-banned-import

from narwhals._plan import expressions as ir

if TYPE_CHECKING:
    from typing_extensions import TypeIs

    from narwhals._plan.arrow.typing import Arrow, ScalarT


def is_arrow(obj: Arrow[ScalarT] | Any) -> TypeIs[Arrow[ScalarT]]:
    return isinstance(obj, (pa.Scalar, pa.Array, pa.ChunkedArray))


class MinMax(ir.AggExpr):
    """Returns a `Struct({'min': ..., 'max': ...})`.

    https://arrow.apache.org/docs/python/generated/pyarrow.compute.min_max.html#pyarrow.compute.min_max
    """
