"""Compatibility module for newer [`enum`] features, like [`StrEnum`].

[`enum`]: https://docs.python.org/3/library/enum.html
[`StrEnum`]: https://docs.python.org/3/library/enum.html#enum.StrEnum
"""

from __future__ import annotations

# ruff: noqa: ARG004
import sys

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from enum import Enum
    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        from typing import Any

        from typing_extensions import Self

    class ReprEnum(Enum): ...

    class StrEnum(str, ReprEnum):
        def __new__(cls, value: str) -> Self:
            if not isinstance(value, str):
                msg = f"{value!r} is not a string"
                raise TypeError(msg)
            member = str.__new__(cls, value)
            member._value_ = value
            return member

        def __str__(self) -> str:
            return str(self.value)

        @staticmethod
        def _generate_next_value_(
            name: str, start: int, count: int, last_values: list[Any]
        ) -> Any:
            return name.lower()


__all__ = ["StrEnum"]
