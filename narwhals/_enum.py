from __future__ import annotations

# ruff: noqa: ARG004
from enum import Enum
from typing import Any


class NoAutoEnum(Enum):
    """Enum base class that prohibits the use of auto() for value assignment.

    This class inherits from enum.Enum but overrides the value generation mechanism
    to prevent automatic value assignment via enum.auto().

    Examples:
        >>> from enum import auto
        >>> from narwhals._enum import NoAutoEnum
        >>>
        >>> class Colors(NoAutoEnum):
        ...     RED = 1
        ...     GREEN = 2
        >>> Colors.RED
        <Colors.RED: 1>

        >>> class ColorsWithAuto(NoAutoEnum):
        ...     RED = 1
        ...     GREEN = auto()
        Traceback (most recent call last):
        ...
        ValueError: Creating values with `auto()` is not allowed. Please provide a value manually instead.

    Raises:
        TypeError: If `auto()` is attempted to be used for any enum member value.

    Note:
        All enum values must be explicitly assigned. This is useful when you want
        to maintain full control over enum values and prevent automatic numbering.

    """

    @staticmethod
    def _generate_next_value_(
        name: str, start: int, count: int, last_values: list[Any]
    ) -> Any:
        msg = "Creating values with `auto()` is not allowed. Please provide a value manually instead."
        raise ValueError(msg)


__all__ = ["NoAutoEnum"]
