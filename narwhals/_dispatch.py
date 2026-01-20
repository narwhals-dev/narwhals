from __future__ import annotations

from collections.abc import Callable
from functools import update_wrapper
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, cast, overload

from narwhals._utils import qualified_type_name

if TYPE_CHECKING:
    from typing_extensions import TypeAlias, Unpack


Incomplete: TypeAlias = Any
R = TypeVar("R")
Fn: TypeAlias = Callable[..., R]


class _JustDispatch(Protocol[R]):
    """The type of a function decorated by `@just_dispatch`."""

    registry: MappingProxyType[type[Any], Fn[R]]
    """Read-only mapping of all registered implementations."""
    dispatch: Callable[[type[Any]], Fn[R]]
    register: Callable[[Unpack[tuple[type[Any], ...]]], Callable[[Fn[R]], Fn[R]]]
    """Register one or more types to dispatch to this function."""

    def __call__(self, /, *args: Incomplete, **kwds: Incomplete) -> R: ...


@overload
def just_dispatch(*, default: type[Any] = ...) -> Callable[[Fn[R]], _JustDispatch[R]]: ...
@overload
def just_dispatch(func: Fn[R], /, *, default: type[Any]) -> _JustDispatch[R]: ...
def just_dispatch(
    func: Fn[R] | None = None, /, *, default: type[Any] = object
) -> _JustDispatch[R] | Fn[_JustDispatch[R]]:
    """An alternative take on [`@singledispatch`].

    - Registering ABCs or anything via type hints is not supported
    - All registered types are dispatched to **by identity**
      - The default implementation can be used for a more relaxed `issubclass` check
      - But this can be constrained via `@just_dispatch(default=...)`

    [`@singledispatch`]: https://docs.python.org/3/library/functools.html#functools.singledispatch
    """
    if func is None:
        return lambda f: just_dispatch(f, default=default)
    registry: dict[type[Any], Fn[R]] = {}

    def dispatch(tp: type[Any], /) -> Fn[R]:
        nonlocal registry
        if f := registry.get(tp):
            return f
        if issubclass(tp, default):
            f = registry[tp] = registry[default]
            return f
        msg = f"{func_name!r} does not support {qualified_type_name(tp)!r} as this is incompatible with default {qualified_type_name(default)!r}"
        raise TypeError(msg)

    def register(*tps: type[Any]) -> Callable[[Fn[R]], Fn[R]]:
        if not tps:
            msg = f"Need at least 1 type to register to {func_name!r}"
            raise TypeError(msg)

        def dec(f: Fn[R], /) -> Fn[R]:
            nonlocal registry
            registry.update((tp, f) for tp in tps)
            return f

        return dec

    def wrapper(*args: Incomplete, **kwds: Incomplete) -> R:
        if not args:
            msg = f"{func_name!r} requires at least 1 positional argument"
            raise TypeError(msg)
        return dispatch(args[0].__class__)(*args, **kwds)

    func_name = func.__name__
    registry[default] = func
    wrapper = cast("_JustDispatch[R]", wrapper)
    wrapper.register = register
    wrapper.dispatch = dispatch
    wrapper.registry = MappingProxyType(registry)
    update_wrapper(wrapper, func)
    return wrapper
