from __future__ import annotations

from collections.abc import Callable
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Generic, TypeVar, overload

from narwhals._utils import qualified_type_name

if TYPE_CHECKING:
    from typing_extensions import TypeAlias


Incomplete: TypeAlias = Any
R = TypeVar("R")
R_co = TypeVar("R_co", covariant=True)
Impl: TypeAlias = Callable[..., R]
Deferred: TypeAlias = Callable[..., R]
Passthrough = TypeVar("Passthrough", bound=Callable[..., Any])


class JustDispatch(Generic[R_co]):
    """The type of a function decorated by `@just_dispatch`."""

    def __init__(self, function: Impl[R_co], /, default: type[Any]) -> None:
        self._function_name: str = function.__name__
        self._default: type[Any] = default
        self._registry: dict[type[Any], Impl[R_co]] = {default: function}
        self.__wrapped__: Impl[R_co] = function

    @property
    def registry(self) -> MappingProxyType[type[Any], Impl[R_co]]:
        """Read-only mapping of all registered implementations."""
        return MappingProxyType(self._registry)

    def dispatch(self, tp: type[Any], /) -> Impl[R_co]:
        """Get the implementation for `tp`."""
        if f := self._registry.get(tp):
            return f
        default = self._default
        if issubclass(tp, default):
            f = self._registry[tp] = self._registry[default]
            return f
        msg = f"{self._function_name!r} does not support {qualified_type_name(tp)!r} as this is incompatible with default {qualified_type_name(default)!r}"
        raise TypeError(msg)

    # TODO @dangotbanned: Turn all these notes into useful docs
    def register(
        self, tp: type[Any], *tps: type[Any]
    ) -> Callable[[Passthrough], Passthrough]:
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

        def decorate(f: Passthrough, /) -> Passthrough:
            if tps:
                self._registry.update((tp_, f) for tp_ in (tp, *tps))
            else:
                self._registry[tp] = f
            return f

        return decorate

    def __call__(self, arg: object, *args: Incomplete, **kwds: Incomplete) -> R_co:
        """Im a doc for everything."""
        return self.dispatch(arg.__class__)(arg, *args, **kwds)


# TODO @dangotbanned: Polish notes/docs
# TODO @dangotbanned: Prefer examples over lots of words
# TODO @dangotbanned: Rename `default` -> `bound`/`upper_bound`
@overload
def just_dispatch(function: Impl[R_co], /) -> JustDispatch[R_co]: ...
@overload
def just_dispatch(
    *, default: type[Any] = object
) -> Callable[[Deferred[R]], JustDispatch[R]]: ...
@overload
def just_dispatch(
    function: Impl[R_co], /, *, default: type[Any]
) -> JustDispatch[R_co]: ...
def just_dispatch(
    function: Impl[R_co] | None = None, /, *, default: type[Any] = object
) -> JustDispatch[R_co] | Callable[[Deferred[R]], JustDispatch[R]]:
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
