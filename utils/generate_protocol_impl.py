"""Tool for stubbing out a new implementation of a protocol."""

from __future__ import annotations

import inspect
import sys
from collections import deque
from typing import TYPE_CHECKING, Literal, TypedDict

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from typing_extensions import LiteralString, TypeAlias

    class Imports(TypedDict):
        runtime: deque[str]
        typing: deque[str]


PlaceholderDescriptor: TypeAlias = Literal["not_implemented", "todo"]

INDENT = " " * 4
NL = "\n"
IMPORT_DESCRIPTOR: Mapping[PlaceholderDescriptor, LiteralString] = {
    "todo": "from narwhals._plan.common import todo",
    "not_implemented": "from narwhals._utils import not_implemented",
}


# TODO @dangotbanned: Major cleaning up
# - TODO @dangotbanned: Structure
#   - [x] At the very least, split the function apart
#   - [x] Even better, use `TypedDict`
#   - [ ] Even better, use classes
# TODO @dangotbanned: Option for running through `ruff`
# - Adapt what was started in `simplify-check-docstrings`
def generate_protocol_impl(
    target_name: LiteralString,
    tp_protocol: type,
    /,
    *,
    placeholder_descriptor: PlaceholderDescriptor = "not_implemented",
    placeholder_alias: LiteralString = "Incomplete",
) -> str:
    """Generate the boilerplate for a new protocol implementation.

    Note:
        Similar idea to the [*Implement all inherited abstract classes*] code action.

    Arguments:
        target_name: Name of the new class
        tp_protocol: Protocol to implement (`typing_extensions.TypeForm[<Protocol?>]`)
        placeholder_descriptor: Name of existing descriptor to mark unimplemented members.
        placeholder_alias: Name of new type alias to mark unimplemented types.

    Returns:
        The module definition as a string.

    [*Implement all inherited abstract classes*]: https://devblogs.microsoft.com/python/python-in-visual-studio-code-may-2024-release/#“implement-all-inherited-abstract-classes”-code-action
    """
    sections: Imports = {"runtime": deque(), "typing": deque()}
    aliases: dict[str, str] = {}
    tp_params_subscript: str = ""

    sections["runtime"].extend(("from __future__ import annotations", ""))
    if params := getattr(tp_protocol, "__parameters__", ()):
        sections["runtime"].append("from typing import TYPE_CHECKING, Any")
        sections["typing"].append("from typing_extensions import TypeAlias")
        aliases[placeholder_alias] = "Any"
        tp_params_subscript = f"[{', '.join((placeholder_alias,) * len(params))}]"
    import_protocol = f"from {tp_protocol.__module__} import {tp_protocol.__name__}"
    descriptor = placeholder_descriptor
    sections["runtime"].extend((import_protocol, IMPORT_DESCRIPTOR[descriptor]))

    if impl := NL.join(_iter_members_lines(tp_protocol, descriptor)):
        impl = f"\n{impl}"
    return (
        f"{NL.join(_iter_import_lines(sections))}\n\n"
        f"{NL.join(_iter_alias_lines(aliases))}\n\n"
        f"class {target_name}({tp_protocol.__name__}{tp_params_subscript}):{impl or ' ...'}"
    )


def _iter_import_lines(sections: Imports) -> Iterator[str]:
    yield from sections["runtime"]
    if section := sections["typing"]:
        yield ""
        yield "if TYPE_CHECKING:"
        yield from (f"{INDENT}{line}" for line in section)


def _iter_alias_lines(aliases: Mapping[str, str]) -> Iterator[str]:
    if aliases:
        yield from (f"{name}: TypeAlias = {value}" for name, value in aliases.items())


def _iter_members_lines(
    tp_protocol: type, placeholder_descriptor: PlaceholderDescriptor
) -> Iterator[str]:
    # inline import so this module is safe to import from anywhere
    if sys.version_info >= (3, 13):
        from typing import get_protocol_members
    else:
        from typing_extensions import get_protocol_members

    members = get_protocol_members(tp_protocol)
    template = f"{INDENT}{{name}} = {placeholder_descriptor}()"
    if members_property := inspect.getmembers(
        tp_protocol, lambda x: isinstance(x, property)
    ):
        prop_names = tuple(name for name, _ in members_property)
        for name in sorted(members.difference(prop_names)):
            yield template.format(name=name)
        yield ""
        for name in prop_names:
            yield f"{INDENT}{name} = {placeholder_descriptor}()  # type: ignore[assignment]"
    else:
        for name in sorted(members):
            yield template.format(name=name)
