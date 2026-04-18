from __future__ import annotations

from typing import TYPE_CHECKING, Any

from narwhals.testing.constructors import (
    EagerFrameConstructor,
    FrameConstructor,
    LazyFrameConstructor,
)

if TYPE_CHECKING:
    from typing_extensions import TypeAlias


__all__ = ("Data", "EagerFrameConstructor", "FrameConstructor", "LazyFrameConstructor")


Data: TypeAlias = dict[str, Any]  # TODO(Unassined): This should have a better annotation
"""A column-oriented mapping used as input to a [`Constructor`][]."""
