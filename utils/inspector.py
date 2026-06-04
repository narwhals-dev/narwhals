"""Introspection utilities we keep rewriting."""

from __future__ import annotations

import inspect
import string
import sys
from inspect import isfunction, ismethoddescriptor
from itertools import groupby
from operator import attrgetter
from types import MethodType
from typing import TYPE_CHECKING, Any, Final, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

LOWERCASE: Final = tuple(string.ascii_lowercase)

T_co = TypeVar("T_co", covariant=True)

GET_MODULE: Callable[[type[Any]], str] = attrgetter("__module__")


def iter_descendants(*bases: type[T_co]) -> Iterator[type[T_co]]:
    """Recursively crawl through subclasses until we stop seeing new things.

    ## Notes
    - This does not de-duplicate classes
    - `__subclasses__` will only return descendants defined in modules that have already been imported
    """
    seen = set[T_co]()
    for base in bases:
        yield base
        if (children := (base.__subclasses__())) and (
            unseen := set(children).difference(seen)
        ):
            yield from iter_descendants(*unseen)


if sys.version_info >= (3, 13):

    def is_public_method_or_property(obj: Any) -> bool:
        return (
            isfunction(obj)
            or (isinstance(obj, (MethodType, property)) or ismethoddescriptor(obj))
        ) and obj.__name__.startswith(LOWERCASE)
else:

    def is_public_method_or_property(obj: Any) -> bool:
        return (
            (isfunction(obj) or (isinstance(obj, MethodType) or ismethoddescriptor(obj)))
            and obj.__name__.startswith(LOWERCASE)
        ) or (isinstance(obj, property) and obj.fget.__name__.startswith(LOWERCASE))


def iter_public_member_names(tp: type[Any]) -> Iterator[str]:
    """Yield the name of anything on `tp` that could be considered public."""
    for name, _ in inspect.getmembers(tp, is_public_method_or_property):
        yield name


def qualified_type_name(tp: type[Any], /) -> str:
    """Return the module-qualified type name.

    This less flexible (but cheaper) version of `narwhals._utils.qualified_type_name`.
    """
    return f"{tp.__module__}.{tp.__name__}"


def group_descendant_names_by_module(*bases: type[T_co]) -> dict[str, list[str]]:
    """Return a mapping from qualified module name to the names of subclasses that were found there."""
    unique_public = {
        tp for tp in iter_descendants(*bases) if not tp.__name__.startswith("_")
    }
    by_module = groupby(sorted(unique_public, key=qualified_type_name), key=GET_MODULE)
    return {k: [tp.__name__ for tp in tps] for k, tps in by_module}
