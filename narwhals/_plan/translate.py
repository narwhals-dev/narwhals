# TODO @dangotbanned: Utilize plugins machinery

from __future__ import annotations

import functools
import threading
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import narwhals.dependencies as deps
from narwhals._plan.plugins._manager import from_native_dataframe, from_native_series
from narwhals._utils import qualified_type_name

if TYPE_CHECKING:
    from collections.abc import Callable

    import polars as pl
    from typing_extensions import TypeAlias, TypeIs

    from narwhals._plan.compliant.lazyframe import CompliantLazyFrame
    from narwhals._plan.compliant.typing import Native as Lazy

    T = TypeVar("T")
    # A lazy type guard like in `narwhals.dependencies` or `narwhals._native`
    _Guard: TypeAlias = "Callable[[Any], TypeIs[T]]"
    _ConstructorLazy: TypeAlias = Callable[[T], CompliantLazyFrame[T]]
    # Zero-argument importer of the concrete native class
    # If we matched a subclass, we would want to avoid registering that on
    # the dispatch function
    _ImportKnown: TypeAlias = Callable[[], type[T]]


__all__ = ("from_native_dataframe", "from_native_lazyframe", "from_native_series")


# TODO @dangotbanned: Replace with `PluginManager` version
@functools.singledispatch
def from_native_lazyframe(native: Lazy, /) -> CompliantLazyFrame[Lazy]:
    if (compliant := _try_known_lazyframes(native)) is not None:
        return compliant
    raise _from_native_error(native, "lazyframe")


def _from_native_error(native: Any, kind: Literal["dataframe", "lazyframe"]) -> TypeError:
    name = qualified_type_name(native)
    msg = f"Unsupported {kind} type, got: {name!r}\n\n{native!r}"
    return TypeError(msg)


def _try_known_lazyframes(native: Lazy, /) -> CompliantLazyFrame[Lazy] | None:
    """Search through known lazyframes and stop if we hit something expected.

    ## Super high-level
    - `@singledispatch` starts with no registered implementations
      - Registration happens after the first match
    - Upon registration, we remove the match from the search space
    - If we see the same type again, it will work the same way as if we did things eagerly

    ## Hairy bits
    - [`threading.local`] is used for safely mutating what *looks like* module-level state
      - the dict is just references to functions, so copies are cheap
    - Lazy registration happens from inside the dispatch function
      - This is fine, provided we don't call the outer dispatch function and land ourselves in a cycle

    [`threading.local`]: https://py-free-threading.github.io/porting/#converting-global-state-to-thread-local-state
    """
    matched: tuple[type[Lazy], _Guard[Lazy], _ConstructorLazy[Lazy]] | None = None
    search: dict[_Guard[Lazy], tuple[_ConstructorLazy[Lazy], _ImportKnown[Lazy]]] = (
        _local.lazy_known
    )
    for guard, (constructor, import_known) in search.items():
        if guard(native):
            matched = (import_known(), guard, constructor)
            break
        else:  # pragma: no cover  # noqa: RET508
            ...
    if matched:
        tp_native, guard, constructor = matched
        from_native_lazyframe.register(tp_native, constructor)
        del _local.lazy_known[guard]
        return constructor(native)
    return None


# TODO @dangotbanned: Review backend/version entrypoint
def _from_polars_lazyframe(native: pl.LazyFrame, /) -> CompliantLazyFrame[pl.LazyFrame]:
    from narwhals._plan.polars import LazyFrame

    return LazyFrame.from_native(native)


def _import_polars_lazyframe() -> type[pl.LazyFrame]:
    import polars as pl  # ignore-banned-import

    return pl.LazyFrame


_local = threading.local()
_local.lazy_known = {
    deps.is_polars_lazyframe: (_from_polars_lazyframe, _import_polars_lazyframe)
}
