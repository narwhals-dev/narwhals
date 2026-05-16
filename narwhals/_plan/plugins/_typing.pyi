# ruff: noqa: PYI021
"""Super secret stub for the hairy bits."""

from collections.abc import Callable
from typing import Any, Generic, ParamSpec, TypeVar, type_check_only

_P = ParamSpec("_P")
_R_co = TypeVar("_R_co", covariant=True)

@type_check_only
class SingleDispatchCallable(Generic[_P, _R_co]):
    def register(self, tp: type[Any], func: Callable[..., Any]) -> Callable[..., Any]: ...
    def __call__(self, *args: _P.args, **kwds: _P.kwargs) -> _R_co: ...

def from_native_dispatch(f: Callable[_P, _R_co], /) -> SingleDispatchCallable[_P, _R_co]:
    """`@functools.singledispatch`, with a parameter preserving signature."""
