"""Introspection utilities we keep rewriting."""

from __future__ import annotations

import inspect
import string
import sys
from inspect import isfunction, ismethoddescriptor
from operator import attrgetter
from types import MethodType
from typing import TYPE_CHECKING, Any, Final, TypeVar

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


T_co = TypeVar("T_co", covariant=True)

LOWERCASE: Final = tuple(string.ascii_lowercase)
NL: Final = "\n"
WS: Final = " "

MEMBERS_START: Final = f"members:{NL}"
MEMBER_PREFIX: Final = f"{WS * 8}-{WS}"

PRIVATE_PREFIX = "_"
DUNDER_PREFIX = "__"

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


def generate_function_autodoc(title: str = "Function") -> str:
    """Collect the meaningful `Function` defs and build a string for mkdocstrings.

    Pretty side-effect heavy, relies on the re-export location having everything in scope.
    """
    from narwhals._plan import expressions as ir

    module, func = "module", "function"
    sources = (
        (tp.__module__, tp.__name__)
        for tp in iter_descendants(ir.Function)
        if not tp.__name__.startswith(PRIVATE_PREFIX)
    )
    df = (
        pl.DataFrame(sources, (module, func))
        .unique()
        .pipe(_sort_functions, module, func, "Function")
        .group_by(module)
        .agg((pl.lit(MEMBER_PREFIX) + pl.col(func)).implode().list.join(NL))
        .select(
            pl.concat_str(
                pl.lit("::: "),
                module,
                pl.lit(f"{NL}{WS * 4}options:{NL}{WS * 6}{MEMBERS_START}"),
                pl.nth(1),
            ).str.join(NL * 2)
        )
    )
    content: str = df.item()
    return f"# {title}{NL * 2}{content}"


def _sort_functions(
    frame: pl.DataFrame, /, col_module: str, col_name: str, base_suffix: str
) -> pl.DataFrame:
    # - Prioritize listing the base class first, if a module has one (e.g. `StringFunction`)
    # - In `_function`, this also breaks ties to order by `Function`, then arity
    name = pl.col(col_name)
    key = pl.when(name.str.ends_with(base_suffix)).then(name.str.len_chars())
    return frame.sort(col_module, key, col_name, nulls_last=True)
