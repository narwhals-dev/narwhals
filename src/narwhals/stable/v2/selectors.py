from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING

from narwhals import selectors as nw_s

if TYPE_CHECKING:
    from collections.abc import Callable

    from typing_extensions import ParamSpec

    from narwhals.stable.v2 import Selector

    P = ParamSpec("P")


def _stableify(fn: Callable[P, nw_s.Selector], /) -> Callable[P, Selector]:
    """Re-wrap a main-namespace selector factory so that it returns a stable `Selector`."""

    @wraps(fn)
    def wrapper(*args: P.args, **kwds: P.kwargs) -> Selector:
        from narwhals.stable.v2 import Selector

        return Selector(*fn(*args, **kwds)._nodes)

    return wrapper


all = _stableify(nw_s.all)
boolean = _stableify(nw_s.boolean)
by_dtype = _stableify(nw_s.by_dtype)
categorical = _stableify(nw_s.categorical)
datetime = _stableify(nw_s.datetime)
enum = _stableify(nw_s.enum)
matches = _stableify(nw_s.matches)
numeric = _stableify(nw_s.numeric)
string = _stableify(nw_s.string)

__all__ = [
    "all",
    "boolean",
    "by_dtype",
    "categorical",
    "datetime",
    "enum",
    "matches",
    "numeric",
    "string",
]
