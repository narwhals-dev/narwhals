"""Backend-specfic type guards."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pyarrow as pa  # ignore-banned-import
import pyarrow.compute as pc  # ignore-banned-import

if TYPE_CHECKING:
    from typing_extensions import TypeIs

    from narwhals._plan.arrow import acero
    from narwhals._plan.arrow.typing import Arrow, ChunkedArrayAny, ScalarT
    from narwhals._utils import _StoresNative

__all__ = ["is_arrow", "is_expression", "is_series"]


def is_series(obj: Any) -> TypeIs[_StoresNative[ChunkedArrayAny]]:
    """Return True if `obj` is a (Compliant) ArrowSeries."""
    from narwhals._plan.arrow.series import ArrowSeries

    return isinstance(obj, ArrowSeries)


def is_arrow(obj: Arrow[ScalarT] | Any) -> TypeIs[Arrow[ScalarT]]:
    """Return True if `obj` is a (Native) Arrow data container."""
    return isinstance(obj, (pa.Scalar, pa.Array, pa.ChunkedArray))


def is_expression(obj: Any) -> TypeIs[acero.Expr]:
    """Return True if `obj` is a (Native) [`Expression`].

    [`Expression`]: https://arrow.apache.org/docs/python/generated/pyarrow.dataset.Expression.html
    """
    return isinstance(obj, pc.Expression)
