from __future__ import annotations

from collections.abc import Callable
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Generic, TypeVar, overload

from narwhals._utils import qualified_type_name

if TYPE_CHECKING:
    from typing_extensions import ParamSpec, TypeAlias

    P = ParamSpec("P")


Incomplete: TypeAlias = Any
R = TypeVar("R")
R_co = TypeVar("R_co", covariant=True)
Fn: TypeAlias = Callable[..., R]
Fn_co: TypeAlias = Callable[..., R_co]
Default = TypeVar("Default", bound=object)


class JustDispatch(Generic[R_co, Default]):
    """The type of a function decorated by `@just_dispatch`."""

    def __init__(self, function: Callable[..., R_co], default: type[Default], /) -> None:
        self._function_name: str = function.__name__
        self._default: type[Default] = default
        self._registry: dict[type[Any], Fn_co[R_co]] = {default: function}
        self.__wrapped__: Callable[..., R_co] = function

    @property
    def registry(self) -> MappingProxyType[type[Any], Callable[..., R_co]]:
        """Read-only mapping of all registered implementations."""
        return MappingProxyType(self._registry)

    def dispatch(self, tp: type[Any], /) -> Fn[R_co]:
        """Get the implementation for `tp`."""
        if f := self._registry.get(tp):
            return f
        default = self._default
        if issubclass(tp, default):
            f = self._registry[tp] = self._registry[default]
            return f
        msg = f"{self._function_name!r} does not support {qualified_type_name(tp)!r} as this is incompatible with default {qualified_type_name(default)!r}"
        raise TypeError(msg)

    def register(
        self, tp: type[Any], *tps: type[Any]
    ) -> Callable[[Callable[P, R_co]], Callable[P, R_co]]:
        """Register one or more types to dispatch to this function.

        Unlike `@singledisptatch`:
        - Registering ABCs or anything via type hints is not supported
        - All registered types are dispatched to **by identity**
        - Multiple types can be registered in a single call
        - Lambda's cannot be used like `function.register(int, lambda x: x + 1)`
            - They should be avoided anyway

        Similar to `@singledisptatch`:
        - This can be used as a decorator
        - The registered function is returned unchanged
        """

        def decorate(f: Callable[P, R_co], /) -> Callable[P, R_co]:
            if tps:
                self._registry.update((tp, f) for tp in tps)
            else:
                self._registry[tp] = f
            return f

        return decorate

    def __call__(self, arg: Default, *args: Incomplete, **kwds: Incomplete) -> R_co:
        """Im a doc for everything."""
        return self.dispatch(arg.__class__)(arg, *args, **kwds)


@overload
def just_dispatch(function: Fn[R], /) -> JustDispatch[R, object]: ...
@overload
def just_dispatch(
    *, default: type[Default] = object
) -> Callable[[Fn[R]], JustDispatch[R, Default]]: ...
@overload
def just_dispatch(
    function: Fn[R], /, *, default: type[Default]
) -> JustDispatch[R, Default]: ...
def just_dispatch(
    function: Fn[R] | None = None, /, *, default: type[Default] = object
) -> JustDispatch[R, Default] | Fn[JustDispatch[R, Default]]:
    """A less dynamic take on [`@functools.singledispatch`].

    Use this if you find yourself creating a global `dict` mapping types to functions.

    Do not use this if you want subclasses to act like they've been registered.

    Arguments:
        function: (Decorating) function to transform into a single dispatch function.
        default: Nominal upper bound to constrain the default implementation.

    Notes:
        - Implements a subset of the api, with a few extras.
        - Most things that are *not implemented* are to improve performance.
        - The result requires less of an understanding of [python's data model] by performing zero magic with [MRO].
        - The `just_*` name is borrowed from [optype - Just]

    [`@functools.singledispatch`]: https://docs.python.org/3/library/functools.html#functools.singledispatch
    [python's data model]: https://docs.python.org/3/reference/datamodel.html
    [MRO]: https://docs.python.org/3/howto/mro.html#python-2-3-mro
    [optype - Just]: https://github.com/jorenham/optype/blob/e7221ed1d3d02989d5d01873323bac9f88459f26/README.md#just
    """
    if function is None:
        return lambda f, /: just_dispatch(f, default=default)
    return JustDispatch(function, default)
