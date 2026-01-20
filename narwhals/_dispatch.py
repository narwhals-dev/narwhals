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
    """Single-dispatch wrapper produced by decorating a function with `@just_dispatch`."""

    __slots__ = ("__wrapped__", "_registry", "_upper_bound")

    def __init__(self, function: Impl[R_co], /, upper_bound: type[Any]) -> None:
        self._upper_bound: type[Any] = upper_bound
        self._registry: dict[type[Any], Impl[R_co]] = {upper_bound: function}
        self.__wrapped__: Impl[R_co] = function

    @property
    def _function_name(self) -> str:
        return self.__wrapped__.__name__

    @property
    def registry(self) -> MappingProxyType[type[Any], Impl[R_co]]:
        """Read-only mapping of all registered implementations."""
        return MappingProxyType(self._registry)

    def dispatch(self, tp: type[Any], /) -> Impl[R_co]:
        """Get the implementation for a given type."""
        if f := self._registry.get(tp):
            return f
        upper = self._upper_bound
        if issubclass(tp, upper):
            f = self._registry[tp] = self._registry[upper]
            return f
        msg = f"{self._function_name!r} does not support {qualified_type_name(tp)!r} as this is incompatible with upper bound {qualified_type_name(upper)!r}"
        raise TypeError(msg)

    # TODO @dangotbanned: Turn all these notes into useful docs
    def register(  # noqa: D417
        self, tp: type[Any], *tps: type[Any]
    ) -> Callable[[Passthrough], Passthrough]:
        """Register types to dispatch via the decorated function.

        Arguments:
            *tps: One or more **concrete** types.

        Returns:
            A closure that can be used as a decorator.

        Notes:
            - Unlike `@singledisptatch`
                - Registering ABCs or anything via type hints is not supported
                - All registered types are dispatched to **by identity**
                - Multiple types can be registered in a single call
                - Lambda's cannot be used like `function.register(int, lambda x: x + 1)`
                    - They should be avoided anyway
            - Similar to `@singledisptatch`
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
        """Dispatch on the type of the first argument, passing through all arguments."""
        return self.dispatch(arg.__class__)(arg, *args, **kwds)


@overload
def just_dispatch(function: Impl[R_co], /) -> JustDispatch[R_co]: ...
@overload
def just_dispatch(
    *, upper_bound: type[Any] = object
) -> Callable[[Deferred[R]], JustDispatch[R]]: ...
@overload
def just_dispatch(
    function: Impl[R_co], /, *, upper_bound: type[Any]
) -> JustDispatch[R_co]: ...
def just_dispatch(
    function: Impl[R_co] | None = None, /, *, upper_bound: type[Any] = object
) -> JustDispatch[R_co] | Callable[[Deferred[R]], JustDispatch[R]]:
    """Transform a function into a single-dispatch generic function.

    Implements a subset of [`@singledispatch`].

    **Do** use this if you find yourself creating a global `dict` mapping types to functions.

    **Do not** use this if you want ABCs to work like [`@singledispatch`].

    Arguments:
        function: Function to decorate, where the body serves as the default implementation.
        upper_bound: When there is no registered implementation for a specific type, require
            the type to be a subclass of `upper_bound` to use the default implementation.

    Notes:
        Most things that are *not implemented* are to give predictable performance,
        and require less of an understanding of [python's data model]. **Do not** expect [MRO] or type annotation magic.

        The `just_*` name is borrowed from [optype - Just].

    [`@singledispatch`]: https://docs.python.org/3/library/functools.html#functools.singledispatch
    [python's data model]: https://docs.python.org/3/reference/datamodel.html
    [MRO]: https://docs.python.org/3/howto/mro.html#python-2-3-mro
    [optype - Just]: https://github.com/jorenham/optype/blob/e7221ed1d3d02989d5d01873323bac9f88459f26/README.md#just
    """
    if function is None:
        return lambda f, /: just_dispatch(f, upper_bound=upper_bound)
    return JustDispatch(function, upper_bound)
