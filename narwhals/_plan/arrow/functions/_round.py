from __future__ import annotations

import typing as t
from typing import TYPE_CHECKING, overload

import pyarrow.compute as pc  # ignore-banned-import

from narwhals._plan.arrow.functions._horizontal import max_horizontal, min_horizontal

if TYPE_CHECKING:
    from narwhals._plan.arrow.typing import (
        ArrowAny,
        ChunkedOrArrayT,
        ChunkedOrScalarAny,
        UnaryNumeric,
    )

__all__ = ["ceil", "clip", "clip_lower", "clip_upper", "floor", "round"]

ceil = t.cast("UnaryNumeric", pc.ceil)
floor = t.cast("UnaryNumeric", pc.floor)


@overload
def round(native: ChunkedOrScalarAny, decimals: int = ...) -> ChunkedOrScalarAny: ...
@overload
def round(native: ChunkedOrArrayT, decimals: int = ...) -> ChunkedOrArrayT: ...
def round(native: ArrowAny, decimals: int = 0) -> ArrowAny:
    """Round underlying floating point data by `decimals` digits."""
    return pc.round(native, decimals, round_mode="half_towards_infinity")


def clip_lower(
    native: ChunkedOrScalarAny, lower: ChunkedOrScalarAny
) -> ChunkedOrScalarAny:
    return max_horizontal(native, lower)


def clip_upper(
    native: ChunkedOrScalarAny, upper: ChunkedOrScalarAny
) -> ChunkedOrScalarAny:
    return min_horizontal(native, upper)


def clip(
    native: ChunkedOrScalarAny, lower: ChunkedOrScalarAny, upper: ChunkedOrScalarAny
) -> ChunkedOrScalarAny:
    return clip_lower(clip_upper(native, upper), lower)
