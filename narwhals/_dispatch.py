from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar, overload

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

    class Deferred(Protocol):
        def __call__(self, f: Impl[R], /) -> JustDispatch[R]: ...


__all__ = ["just_dispatch"]

R = TypeVar("R")
R_co = TypeVar("R_co", covariant=True)
Impl: TypeAlias = Callable[..., R]
Passthrough = TypeVar("Passthrough", bound=Callable[..., Any])


class JustDispatch(Generic[R_co]):
    """Single-dispatch wrapper produced by decorating a function with `@just_dispatch`."""

    __slots__ = ("_registry", "_upper_bound")

    def __init__(self, function: Impl[R_co], /, upper_bound: type[Any]) -> None:
        self._upper_bound: type[Any] = upper_bound
        self._registry: dict[type[Any], Impl[R_co]] = {upper_bound: function}

    def dispatch(self, tp: type[Any], /) -> Impl[R_co]:
        """Get the implementation for a given type."""
        if f := self._registry.get(tp):
            return f
        if issubclass(tp, self._upper_bound):
            f = self._registry[tp] = self._registry[self._upper_bound]
            return f
        msg = f"{self._registry[self._upper_bound].__name__!r} does not support {tp.__name__!r}"
        raise TypeError(msg)

    def register(
        self, tp: type[Any], *tps: type[Any]
    ) -> Callable[[Passthrough], Passthrough]:
        """Register types to dispatch via the decorated function."""

        def decorate(f: Passthrough, /) -> Passthrough:
            self._registry.update((tp_, f) for tp_ in (tp, *tps))
            return f

        return decorate

    def __call__(self, arg: object, *args: Any, **kwds: Any) -> R_co:
        """Dispatch on the type of the first argument, passing through all arguments."""
        return self.dispatch(arg.__class__)(arg, *args, **kwds)


@overload
def just_dispatch(function: Impl[R], /) -> JustDispatch[R]: ...
@overload
def just_dispatch(*, upper_bound: type[Any] = object) -> Deferred: ...
def just_dispatch(
    function: Impl[R] | None = None, /, *, upper_bound: type[Any] = object
) -> JustDispatch[R] | Deferred:
    if function is not None:
        return JustDispatch(function, upper_bound)
    return partial(JustDispatch[Any], upper_bound=upper_bound)
