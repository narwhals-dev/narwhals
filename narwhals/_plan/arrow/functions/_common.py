"""Things that still need a home."""

from __future__ import annotations

import math
import typing as t
from typing import TYPE_CHECKING, Any

import pyarrow as pa  # ignore-banned-import
import pyarrow.compute as pc  # ignore-banned-import

from narwhals._plan import expressions as ir
from narwhals._plan.arrow.functions._categorical import dictionary_encode
from narwhals._plan.arrow.functions._construction import chunked_array, lit

if TYPE_CHECKING:
    from typing_extensions import TypeIs

    from narwhals._plan.arrow.typing import (
        Arrow,
        ChunkedArrayAny,
        ChunkedOrScalarAny,
        ScalarT,
        UnaryNumeric,
    )

__all__ = ["MinMax", "abs", "exp", "is_arrow", "log", "mode_all"]

abs = t.cast("UnaryNumeric", pc.abs)
exp = t.cast("UnaryNumeric", pc.exp)


def mode_all(native: ChunkedArrayAny) -> ChunkedArrayAny:
    struct_arr = pc.mode(native, n=len(native))
    indices = dictionary_encode(struct_arr.field("count"))
    index_true_modes = lit(0)
    return chunked_array(
        struct_arr.field("mode").filter(pc.equal(indices, index_true_modes))
    )


def log(native: ChunkedOrScalarAny, base: float = math.e) -> ChunkedOrScalarAny:
    return t.cast("ChunkedOrScalarAny", pc.logb(native, lit(base)))


def is_arrow(obj: Arrow[ScalarT] | Any) -> TypeIs[Arrow[ScalarT]]:
    return isinstance(obj, (pa.Scalar, pa.Array, pa.ChunkedArray))


class MinMax(ir.AggExpr):
    """Returns a `Struct({'min': ..., 'max': ...})`.

    https://arrow.apache.org/docs/python/generated/pyarrow.compute.min_max.html#pyarrow.compute.min_max
    """
