"""Backward compatibility for newer/less buggy typing features.

## Important
Import from here to avoid introducing a runtime dependency on [`typing_extensions`]

## Notes
- `TypeVar` defaults
  - https://typing.python.org/en/latest/spec/generics.html#type-parameter-defaults
  - https://peps.python.org/pep-0696/
- `@deprecated`
  - https://docs.python.org/3/library/warnings.html#warnings.deprecated
  - https://typing.python.org/en/latest/spec/directives.html#deprecated
  - https://peps.python.org/pep-0702/

[`typing_extensions`]: https://github.com/python/typing_extensions
"""

from __future__ import annotations

# ruff: noqa: ARG001, ANN202, N802
import sys
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    if sys.version_info >= (3, 13):
        from typing import TypeVar
        from warnings import deprecated
    else:
        from typing_extensions import TypeVar, deprecated

    _Fn = TypeVar("_Fn", bound=Callable[..., Any])


elif sys.version_info >= (3, 13):
    from typing import TypeVar
    from warnings import deprecated
else:
    from typing import TypeVar as _TypeVar

    def TypeVar(
        name: str,
        *constraints: Any,
        bound: Any | None = None,
        covariant: bool = False,
        contravariant: bool = False,
        **kwds: Any,
    ):
        return _TypeVar(
            name,
            *constraints,
            bound=bound,
            covariant=covariant,
            contravariant=contravariant,
        )

    def deprecated(message: str, /) -> Callable[[_Fn], _Fn]:
        def wrapper(func: _Fn, /) -> _Fn:
            return func

        return wrapper


__all__ = ["TypeVar", "deprecated"]
