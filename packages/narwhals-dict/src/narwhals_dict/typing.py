from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any, TypeAlias

    DictFrame: TypeAlias = "dict[str, Sequence[Any]]"
    """Native frame: a mapping from column name to a homogeneous sequence of values."""

    NativeSeries: TypeAlias = "Sequence[Any]"
    """Native column: any non-string sequence with consistent element type, `None` is null."""

    Incomplete: TypeAlias = "Any"

__all__ = ["DictFrame", "Incomplete", "NativeSeries"]
