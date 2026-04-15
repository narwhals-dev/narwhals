from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

    from narwhals._native import NativeLazyFrame
    from narwhals.typing import IntoDataFrame

Data: TypeAlias = dict[str, list[Any]]
"""A column-oriented mapping used as input to a [`Constructor`][]."""

Constructor: TypeAlias = Callable[[Data], "NativeLazyFrame | IntoDataFrame"]
"""Any constructor (eager or lazy) — anything callable that returns a native frame."""

ConstructorEager: TypeAlias = Callable[[Data], "IntoDataFrame"]
"""A constructor that returns an eager native dataframe."""

ConstructorLazy: TypeAlias = Callable[[Data], "NativeLazyFrame"]
"""A constructor that returns a lazy native frame."""

__all__ = ["Constructor", "ConstructorEager", "ConstructorLazy", "Data"]
