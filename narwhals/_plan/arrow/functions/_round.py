"""Round underlying floating point data.

This group is derived from the (rust) polars [feature] [`round_series`].

[feature]: https://docs.rs/polars/latest/polars/#compile-times-and-opt-in-features
[`round_series`]: https://github.com/search?q=repo%3Apola-rs%2Fpolars+path%3A%2F%5Ecrates%5C%2Fpolars-plan%5C%2Fsrc%5C%2Fdsl%5C%2F%2F+%23%5Bcfg%28feature+%3D+%22round_series%22%29%5D&type=code
"""

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
"""Rounds up to the nearest integer value."""
floor = t.cast("UnaryNumeric", pc.floor)
"""Rounds down to the nearest integer value."""


@overload
def round(native: ChunkedOrScalarAny, decimals: int = 0) -> ChunkedOrScalarAny: ...
@overload
def round(native: ChunkedOrArrayT, decimals: int = 0) -> ChunkedOrArrayT: ...
def round(native: ArrowAny, decimals: int = 0) -> ArrowAny:
    """Round underlying floating point data by `decimals` digits."""
    return pc.round(native, decimals, round_mode="half_towards_infinity")


def clip_lower(
    native: ChunkedOrScalarAny, lower: ChunkedOrScalarAny
) -> ChunkedOrScalarAny:
    """Limit values to at-least `lower`."""
    return max_horizontal(native, lower)


def clip_upper(
    native: ChunkedOrScalarAny, upper: ChunkedOrScalarAny
) -> ChunkedOrScalarAny:
    """Limit values to at-most `upper`."""
    return min_horizontal(native, upper)


def clip(
    native: ChunkedOrScalarAny, lower: ChunkedOrScalarAny, upper: ChunkedOrScalarAny
) -> ChunkedOrScalarAny:
    """Set values outside the given boundaries to the boundary value."""
    return clip_lower(clip_upper(native, upper), lower)
