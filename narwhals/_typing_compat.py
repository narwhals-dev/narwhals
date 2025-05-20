from __future__ import annotations

# ruff: noqa: ARG001, ANN202, N802
import sys
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from typing import Callable
    from typing import Protocol as Protocol38

    if sys.version_info >= (3, 13):
        from typing import TypeVar
        from warnings import deprecated
    else:
        from typing_extensions import TypeVar
        from typing_extensions import deprecated

    _Fn = TypeVar("_Fn", bound=Callable[..., Any])


else:  # pragma: no cover
    if sys.version_info >= (3, 13):
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

    # TODO @dangotbanned: Remove after dropping `3.8` (#2084)
    # - https://github.com/narwhals-dev/narwhals/pull/2064#discussion_r1965921386
    if sys.version_info >= (3, 9):
        from typing import Protocol as Protocol38
    else:
        from typing import Generic as Protocol38


__all__ = ["Protocol38", "TypeVar", "deprecated"]
