"""This module implements rendering utilities."""

from __future__ import annotations

import enum
import random
import re
import string
import sys
import warnings
from functools import lru_cache, partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Match, Pattern, Sequence

from griffe.dataclasses import Alias, Object
from griffe.docstrings.dataclasses import (
    DocstringSectionAttributes,
    DocstringSectionClasses,
    DocstringSectionFunctions,
    DocstringSectionModules,
)
from jinja2 import TemplateNotFound, pass_context, pass_environment
from markupsafe import Markup
from mkdocstrings.loggers import get_logger

if TYPE_CHECKING:
    from griffe.dataclasses import Attribute, Class, Function, Module
    from jinja2 import Environment, Template
    from jinja2.runtime import Context
    from mkdocstrings.handlers.base import CollectorItem

logger = get_logger(__name__)


class Order(enum.Enum):
    """Enumeration for the possible members ordering."""

    alphabetical = "alphabetical"
    """Alphabetical order."""
    source = "source"
    """Source code order."""


def _sort_key_alphabetical(item: CollectorItem) -> Any:
    # chr(sys.maxunicode) is a string that contains the final unicode
    # character, so if 'name' isn't found on the object, the item will go to
    # the end of the list.
    return item.name or chr(sys.maxunicode)


def _sort_key_source(item: CollectorItem) -> Any:
    # if 'lineno' is none, the item will go to the start of the list.
    if item.is_alias:
        return item.alias_lineno if item.alias_lineno is not None else -1
    return item.lineno if item.lineno is not None else -1


order_map = {
    Order.alphabetical: _sort_key_alphabetical,
    Order.source: _sort_key_source,
}


def do_format_code(code: str, line_length: int) -> str:
    """Format code using Black.

    Parameters:
        code: The code to format.
        line_length: The line length to give to Black.

    Returns:
        The same code, formatted.
    """
    code = code.strip()
    if len(code) < line_length:
        return code
    formatter = _get_black_formatter()
    return formatter(code, line_length)


_stash_key_alphabet = string.ascii_letters + string.digits


def _gen_key(length: int) -> str:
    return "_" + "".join(random.choice(_stash_key_alphabet) for _ in range(max(1, length - 1)))  # noqa: S311


def _gen_stash_key(stash: dict[str, str], length: int) -> str:
    key = _gen_key(length)
    while key in stash:
        key = _gen_key(length)
    return key


def _stash_crossref(stash: dict[str, str], crossref: str, *, length: int) -> str:
    key = _gen_stash_key(stash, length)
    stash[key] = crossref
    return key


def _format_signature(name: Markup, signature: str, line_length: int) -> str:
    name = str(name).strip()  # type: ignore[assignment]
    signature = signature.strip()
    if len(name + signature) < line_length:
        return name + signature

    # Black cannot format names with dots, so we replace
    # the whole name with a string of equal length
    name_length = len(name)
    formatter = _get_black_formatter()
    formatable = f"def {'x' * name_length}{signature}: pass"
    formatted = formatter(formatable, line_length)

    # We put back the original name
    # and remove starting `def ` and trailing `: pass`
    return name + formatted[4:-5].strip()[name_length:-1]


@pass_context
def do_format_signature(
    context: Context,
    callable_path: Markup,
    function: Function,
    line_length: int,
    *,
    annotations: bool | None = None,
    crossrefs: bool = False,
) -> str:
    """Format a signature using Black.

    Parameters:
        context: Jinja context, passed automatically.
        callable_path: The path of the callable we render the signature of.
        function: The function we render the signature of.
        line_length: The line length to give to Black.
        annotations: Whether to show type annotations.
        crossrefs: Whether to cross-reference types in the signature.

    Returns:
        The same code, formatted.
    """
    env = context.environment
    # TODO: Stop using `do_get_template` when `*.html` templates are removed.
    template = env.get_template(do_get_template(env, "signature"))
    config_annotations = context.parent["config"]["show_signature_annotations"]
    old_stash_ref_filter = env.filters["stash_crossref"]

    stash: dict[str, str] = {}
    if (annotations or config_annotations) and crossrefs:
        env.filters["stash_crossref"] = partial(_stash_crossref, stash)

    if annotations is None:
        new_context = context.parent
    else:
        new_context = dict(context.parent)
        new_context["config"] = dict(new_context["config"])
        new_context["config"]["show_signature_annotations"] = annotations
    try:
        signature = template.render(new_context, function=function, signature=True)
    finally:
        env.filters["stash_crossref"] = old_stash_ref_filter

    signature = _format_signature(callable_path, signature, line_length)
    signature = str(
        env.filters["highlight"](
            Markup.escape(signature),
            language="python",
            inline=False,
            classes=["doc-signature"],
        ),
    )

    # Since we highlight the signature without `def`,
    # Pygments sees it as a function call and not a function definition.
    # The result is that the function name is not parsed as such,
    # but instead as a regular name: `n` CSS class instead of `nf`.
    # To fix it, we replace the first occurrence of an `n` CSS class
    # with an `nf` one, unless we found `nf` already.
    if signature.find('class="nf"') == -1:
        signature = signature.replace('class="n"', 'class="nf"', 1)

    if stash:
        for key, value in stash.items():
            signature = re.sub(rf"\b{key}\b", value, signature)

    return signature


@pass_context
def do_format_attribute(
    context: Context,
    attribute_path: Markup,
    attribute: Attribute,
    line_length: int,
    *,
    crossrefs: bool = False,
) -> str:
    """Format an attribute using Black.

    Parameters:
        context: Jinja context, passed automatically.
        attribute_path: The path of the callable we render the signature of.
        attribute: The attribute we render the signature of.
        line_length: The line length to give to Black.
        crossrefs: Whether to cross-reference types in the signature.

    Returns:
        The same code, formatted.
    """
    env = context.environment
    # TODO: Stop using `do_get_template` when `*.html` templates are removed.
    template = env.get_template(do_get_template(env, "expression"))
    annotations = context.parent["config"]["show_signature_annotations"]
    separate_signature = context.parent["config"]["separate_signature"]
    old_stash_ref_filter = env.filters["stash_crossref"]

    stash: dict[str, str] = {}
    if separate_signature and crossrefs:
        env.filters["stash_crossref"] = partial(_stash_crossref, stash)

    try:
        signature = str(attribute_path).strip()
        if annotations and attribute.annotation:
            annotation = template.render(context.parent, expression=attribute.annotation, signature=True)
            signature += f": {annotation}"
        if attribute.value:
            value = template.render(context.parent, expression=attribute.value, signature=True)
            signature += f" = {value}"
    finally:
        env.filters["stash_crossref"] = old_stash_ref_filter

    signature = do_format_code(signature, line_length)
    signature = str(
        env.filters["highlight"](
            Markup.escape(signature),
            language="python",
            inline=False,
            classes=["doc-signature"],
        ),
    )

    if stash:
        for key, value in stash.items():
            signature = re.sub(rf"\b{key}\b", value, signature)

    return signature


def do_order_members(
    members: Sequence[Object | Alias],
    order: Order,
    members_list: bool | list[str] | None,
) -> Sequence[Object | Alias]:
    """Order members given an ordering method.

    Parameters:
        members: The members to order.
        order: The ordering method.
        members_list: An optional member list (manual ordering).

    Returns:
        The same members, ordered.
    """
    if isinstance(members_list, list) and members_list:
        sorted_members = []
        members_dict = {member.name: member for member in members}
        for name in members_list:
            if name in members_dict:
                sorted_members.append(members_dict[name])
        return sorted_members
    return sorted(members, key=order_map[order])


@lru_cache
def _warn_crossref() -> None:
    warnings.warn(
        "The `crossref` filter is deprecated and will be removed in a future version",
        DeprecationWarning,
        stacklevel=1,
    )


def do_crossref(path: str, *, brief: bool = True) -> Markup:
    """Deprecated. Filter to create cross-references.

    Parameters:
        path: The path to link to.
        brief: Show only the last part of the path, add full path as hover.

    Returns:
        Markup text.
    """
    _warn_crossref()
    full_path = path
    if brief:
        path = full_path.split(".")[-1]
    return Markup("<span data-autorefs-optional-hover={full_path}>{path}</span>").format(full_path=full_path, path=path)


@lru_cache
def _warn_multi_crossref() -> None:
    warnings.warn(
        "The `multi_crossref` filter is deprecated and will be removed in a future version",
        DeprecationWarning,
        stacklevel=1,
    )


def do_multi_crossref(text: str, *, code: bool = True) -> Markup:
    """Deprecated. Filter to create cross-references.

    Parameters:
        text: The text to scan.
        code: Whether to wrap the result in a code tag.

    Returns:
        Markup text.
    """
    _warn_multi_crossref()
    group_number = 0
    variables = {}

    def repl(match: Match) -> str:
        nonlocal group_number
        group_number += 1
        path = match.group()
        path_var = f"path{group_number}"
        variables[path_var] = path
        return f"<span data-autorefs-optional-hover={{{path_var}}}>{{{path_var}}}</span>"

    text = re.sub(r"([\w.]+)", repl, text)
    if code:
        text = f"<code>{text}</code>"
    return Markup(text).format(**variables)


def do_split_path(path: str, full_path: str) -> list[tuple[str, str]]:
    """Split object paths for building cross-references.

    Parameters:
        path: The path to split.

    Returns:
        A list of pairs (title, full path).
    """
    if "." not in path:
        return [(path, full_path)]
    pairs = []
    full_path = ""
    for part in path.split("."):
        if full_path:
            full_path += f".{part}"
        else:
            full_path = part
        pairs.append((part, full_path))
    return pairs


def _keep_object(name: str, filters: Sequence[tuple[Pattern, bool]]) -> bool:
    keep = None
    rules = set()
    for regex, exclude in filters:
        rules.add(exclude)
        if regex.search(name):
            keep = not exclude
    if keep is None:
        if rules == {False}:
            # only included stuff, no match = reject
            return False
        # only excluded stuff, or included and excluded stuff, no match = keep
        return True
    return keep


def do_filter_objects(
    objects_dictionary: dict[str, Object | Alias],
    *,
    filters: Sequence[tuple[Pattern, bool]] | None = None,
    members_list: bool | list[str] | None = None,
    inherited_members: bool | list[str] = False,
    keep_no_docstrings: bool = True,
) -> list[Object | Alias]:
    """Filter a dictionary of objects based on their docstrings.

    Parameters:
        objects_dictionary: The dictionary of objects.
        filters: Filters to apply, based on members' names.
            Each element is a tuple: a pattern, and a boolean indicating whether
            to reject the object if the pattern matches.
        members_list: An optional, explicit list of members to keep.
            When given and empty, return an empty list.
            When given and not empty, ignore filters and docstrings presence/absence.
        inherited_members: Whether to keep inherited members or exclude them.
        keep_no_docstrings: Whether to keep objects with no/empty docstrings (recursive check).

    Returns:
        A list of objects.
    """
    inherited_members_specified = False
    if inherited_members is True:
        # Include all inherited members.
        objects = list(objects_dictionary.values())
    elif inherited_members is False:
        # Include no inherited members.
        objects = [obj for obj in objects_dictionary.values() if not obj.inherited]
    else:
        # Include specific inherited members.
        inherited_members_specified = True
        objects = [
            obj for obj in objects_dictionary.values() if not obj.inherited or obj.name in set(inherited_members)
        ]

    if members_list is True:
        # Return all pre-selected members.
        return objects

    if members_list is False or members_list == []:
        # Return selected inherited members, if any.
        return [obj for obj in objects if obj.inherited]

    if members_list is not None:
        # Return selected members (keeping any pre-selected inherited members).
        return [
            obj for obj in objects if obj.name in set(members_list) or (inherited_members_specified and obj.inherited)
        ]

    # Use filters and docstrings.
    if filters:
        objects = [
            obj for obj in objects if _keep_object(obj.name, filters) or (inherited_members_specified and obj.inherited)
        ]
    if keep_no_docstrings:
        return objects

    return [obj for obj in objects if obj.has_docstrings or (inherited_members_specified and obj.inherited)]


@lru_cache(maxsize=1)
def _get_black_formatter() -> Callable[[str, int], str]:
    try:
        from black import InvalidInput, Mode, format_str
    except ModuleNotFoundError:
        logger.info("Formatting signatures requires Black to be installed.")
        return lambda text, _: text

    def formatter(code: str, line_length: int) -> str:
        mode = Mode(line_length=line_length)
        try:
            return format_str(code, mode=mode)
        except InvalidInput:
            return code

    return formatter


@pass_environment
def do_get_template(env: Environment, obj: str | Object) -> str | Template:
    """Get the template name used to render an object.

    Parameters:
        env: The Jinja environment, passed automatically.
        obj: A Griffe object, or a template name.

    Returns:
        A template name.
    """
    name = obj
    if isinstance(obj, (Alias, Object)):
        extra_data = getattr(obj, "extra", {}).get("mkdocstrings", {})
        if name := extra_data.get("template", ""):
            return name
        name = obj.kind.value
    try:
        template = env.get_template(f"{name}.html")
    except TemplateNotFound:
        return f"{name}.html.jinja"
    # TODO: Remove once support for Python 3.8 is dropped.
    if sys.version_info < (3, 9):
        try:
            Path(template.filename).relative_to(Path(__file__).parent)  # type: ignore[arg-type]
        except ValueError:
            our_template = False
        else:
            our_template = True
    else:
        our_template = Path(template.filename).is_relative_to(Path(__file__).parent)  # type: ignore[arg-type]
    if our_template:
        return f"{name}.html.jinja"
    # TODO: Switch to a warning log after some time.
    logger.info(
        f"DeprecationWarning: Overriding '{name}.html' is deprecated, override '{name}.html.jinja' instead. "
        "After some time, this message will be logged as a warning, causing strict builds to fail.",
        once=True,
    )
    return f"{name}.html"


@pass_context
def do_as_attributes_section(
    context: Context,  # noqa: ARG001
    attributes: Sequence[Attribute],  # noqa: ARG001
    *,
    check_public: bool = True,  # noqa: ARG001
) -> DocstringSectionAttributes:
    """Build an attributes section from a list of attributes.

    Parameters:
        attributes: The attributes to build the section from.
        check_public: Whether to check if the attribute is public.

    Returns:
        An attributes docstring section.
    """
    return DocstringSectionAttributes([])


@pass_context
def do_as_functions_section(
    context: Context,  # noqa: ARG001
    functions: Sequence[Function],  # noqa: ARG001
    *,
    check_public: bool = True,  # noqa: ARG001
) -> DocstringSectionFunctions:
    """Build a functions section from a list of functions.

    Parameters:
        functions: The functions to build the section from.
        check_public: Whether to check if the function is public.

    Returns:
        A functions docstring section.
    """
    return DocstringSectionFunctions([])


@pass_context
def do_as_classes_section(
    context: Context,  # noqa: ARG001
    classes: Sequence[Class],  # noqa: ARG001
    *,
    check_public: bool = True,  # noqa: ARG001
) -> DocstringSectionClasses:
    """Build a classes section from a list of classes.

    Parameters:
        classes: The classes to build the section from.
        check_public: Whether to check if the class is public.

    Returns:
        A classes docstring section.
    """
    return DocstringSectionClasses([])


@pass_context
def do_as_modules_section(
    context: Context,  # noqa: ARG001
    modules: Sequence[Module],  # noqa: ARG001
    *,
    check_public: bool = True,  # noqa: ARG001
) -> DocstringSectionModules:
    """Build a modules section from a list of modules.

    Parameters:
        modules: The modules to build the section from.
        check_public: Whether to check if the module is public.

    Returns:
        A modules docstring section.
    """
    return DocstringSectionModules([])
