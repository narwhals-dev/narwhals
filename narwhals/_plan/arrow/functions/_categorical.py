"""Categorical function namespace."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pyarrow as pa  # ignore-banned-import

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

    Incomplete: TypeAlias = Any

    from narwhals._plan.arrow.typing import ArrowAny, ChunkedArrayAny


__all__ = ["get_categories"]


def get_categories(native: ArrowAny, /) -> ChunkedArrayAny:
    da: Incomplete
    if isinstance(native, pa.ChunkedArray):
        da = native.unify_dictionaries().chunk(0)
    else:
        da = native
    return pa.chunked_array([da.dictionary])
