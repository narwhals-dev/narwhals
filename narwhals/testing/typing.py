from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

    from narwhals.testing.constructors._classes import (
        ConstructorBase,
        ConstructorEagerBase,
        ConstructorLazyBase,
    )

Data: TypeAlias = dict[str, Any]  # TODO(Unassined): This should have a better annotation
"""A column-oriented mapping used as input to a [`Constructor`][]."""

Constructor: TypeAlias = "ConstructorBase"
"""Any constructor (eager or lazy): callable that returns a native frame."""

ConstructorEager: TypeAlias = "ConstructorEagerBase"
"""A constructor that returns an eager native dataframe."""

ConstructorLazy: TypeAlias = "ConstructorLazyBase"
"""A constructor that returns a lazy native frame."""

__all__ = ["Constructor", "ConstructorEager", "ConstructorLazy", "Data"]
