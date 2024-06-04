"""Markdown Exec package.

Utilities to execute code blocks in Markdown files.
"""

# https://facelessuser.github.io/pymdown-extensions/extensions/superfences/#custom-fences
# https://squidfunk.github.io/mkdocs-material/setup/extensions/python-markdown-extensions/#snippets

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from markdown import Markdown

from markdown_exec.formatters.base import default_tabs
from markdown_exec.formatters.bash import _format_bash
from markdown_exec.formatters.console import _format_console
from markdown_exec.formatters.markdown import _format_markdown
from markdown_exec.formatters.pycon import _format_pycon
from markdown_exec.formatters.pyodide import _format_pyodide
from markdown_exec.formatters.python import _format_python
from markdown_exec.formatters.sh import _format_sh
from markdown_exec.formatters.tree import _format_tree

__all__: list[str] = ["formatter", "validator"]


formatters = {
    "bash": _format_bash,
    "console": _format_console,
    "md": _format_markdown,
    "markdown": _format_markdown,
    "py": _format_python,
    "python": _format_python,
    "pycon": _format_pycon,
    "pyodide": _format_pyodide,
    "sh": _format_sh,
    "tree": _format_tree,
}

# negative look behind: matches only if | (pipe) if not preceded by \ (backslash)
_tabs_re = re.compile(r"(?<!\\)\|")


def validator(
    language: str,
    inputs: dict[str, str],
    options: dict[str, Any],
    attrs: dict[str, Any],  # noqa: ARG001
    md: Markdown,  # noqa: ARG001
) -> bool:
    """Validate code blocks inputs.

    Parameters:
        language: The code language, like python or bash.
        inputs: The code block inputs, to be sorted into options and attrs.
        options: The container for options.
        attrs: The container for attrs:
        md: The Markdown instance.

    Returns:
        Success or not.
    """
    exec_value = _to_bool(inputs.pop("exec", "no"))
    if language not in {"tree", "pyodide"} and not exec_value:
        return False
    id_value = inputs.pop("id", "")
    id_prefix_value = inputs.pop("idprefix", None)
    html_value = _to_bool(inputs.pop("html", "no"))
    source_value = inputs.pop("source", "")
    result_value = inputs.pop("result", "")
    returncode_value = int(inputs.pop("returncode", "0"))
    session_value = inputs.pop("session", "")
    update_toc_value = _to_bool(inputs.pop("updatetoc", "yes"))
    tabs_value = inputs.pop("tabs", "|".join(default_tabs))
    tabs = tuple(_tabs_re.split(tabs_value, maxsplit=1))
    options["id"] = id_value
    options["id_prefix"] = id_prefix_value
    options["html"] = html_value
    options["source"] = source_value
    options["result"] = result_value
    options["returncode"] = returncode_value
    options["session"] = session_value
    options["update_toc"] = update_toc_value
    options["tabs"] = tabs
    options["extra"] = inputs
    return True


def formatter(
    source: str,
    language: str,
    css_class: str,  # noqa: ARG001
    options: dict[str, Any],
    md: Markdown,
    classes: list[str] | None = None,  # noqa: ARG001
    id_value: str = "",  # noqa: ARG001
    attrs: dict[str, Any] | None = None,  # noqa: ARG001
    **kwargs: Any,  # noqa: ARG001
) -> str:
    """Execute code and return HTML.

    Parameters:
        source: The code to execute.
        language: The code language, like python or bash.
        css_class: The CSS class to add to the HTML element.
        options: The container for options.
        attrs: The container for attrs:
        md: The Markdown instance.
        classes: Additional CSS classes.
        id_value: An optional HTML id.
        attrs: Additional attributes
        **kwargs: Additional arguments passed to SuperFences default formatters.

    Returns:
        HTML contents.
    """
    fmt = formatters.get(language, lambda source, **kwargs: source)
    return fmt(code=source, md=md, **options)  # type: ignore[operator]


falsy_values = {"", "no", "off", "false", "0"}
truthy_values = {"yes", "on", "true", "1"}


def _to_bool(value: str) -> bool:
    return value.lower() not in falsy_values


def _to_bool_or_value(value: str) -> bool | str:
    if value.lower() in falsy_values:
        return False
    if value.lower() in truthy_values:
        return True
    return value
