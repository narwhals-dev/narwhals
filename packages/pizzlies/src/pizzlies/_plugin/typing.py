from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any, TypeAlias

    from narwhals.utils import Implementation, Version

    DictFrame: TypeAlias = "dict[str, Sequence[Any]]"
    """Native frame: a mapping from column name to a homogeneous sequence of values."""

    NativeSeries: TypeAlias = "Sequence[Any]"
    """Native column: any non-string sequence with consistent element type, `None` is null."""


class Context(Protocol):
    """Anything which stores an implementation and version, e.g. a compliant object."""

    _implementation: Implementation
    _version: Version


__all__ = ["Context", "DictFrame", "NativeSeries"]
