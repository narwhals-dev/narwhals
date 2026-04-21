from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

    from narwhals.testing.constructors import frame_constructor
    from narwhals.typing import IntoDataFrame, IntoFrame, IntoLazyFrame


__all__ = ("Data", "EagerFrameConstructor", "FrameConstructor", "LazyFrameConstructor")

FrameConstructor: TypeAlias = "frame_constructor[IntoFrame]"

EagerFrameConstructor: TypeAlias = "frame_constructor[IntoDataFrame]"
"""Type alias for a constructor that returns an eager native dataframe."""

LazyFrameConstructor: TypeAlias = "frame_constructor[IntoLazyFrame]"
"""Type alias for a constructor that returns a lazy native frame."""


Data: TypeAlias = dict[str, Any]  # TODO(Unassined): This should have a better annotation
"""A column-oriented mapping used as input to a frame constructor."""
