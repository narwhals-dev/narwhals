"""Generic formatter for executing code."""

from __future__ import annotations

from textwrap import indent
from typing import TYPE_CHECKING, Any, Callable
from uuid import uuid4

from markupsafe import Markup

from markdown_exec.logger import get_logger
from markdown_exec.rendering import MarkdownConverter, add_source, code_block

if TYPE_CHECKING:
    from markdown.core import Markdown

logger = get_logger(__name__)
default_tabs = ("Source", "Result")


class ExecutionError(Exception):
    """Exception raised for errors during execution of a code block.

    Attributes:
        message: The exception message.
        returncode: The code returned by the execution of the code block.
    """

    def __init__(self, message: str, returncode: int | None = None) -> None:  # noqa: D107
        super().__init__(message)
        self.returncode = returncode


def _format_log_details(details: str, *, strip_fences: bool = False) -> str:
    if strip_fences:
        lines = details.split("\n")
        if lines[0].startswith("```") and lines[-1].startswith("```"):
            details = "\n".join(lines[1:-1])
    return indent(details, " " * 2)


def base_format(
    *,
    language: str,
    run: Callable,
    code: str,
    md: Markdown,
    html: bool = False,
    source: str = "",
    result: str = "",
    tabs: tuple[str, str] = default_tabs,
    id: str = "",  # noqa: A002
    id_prefix: str | None = None,
    returncode: int = 0,
    transform_source: Callable[[str], tuple[str, str]] | None = None,
    session: str | None = None,
    update_toc: bool = True,
    **options: Any,
) -> Markup:
    """Execute code and return HTML.

    Parameters:
        language: The code language.
        run: Function that runs code and returns output.
        code: The code to execute.
        md: The Markdown instance.
        html: Whether to inject output as HTML directly, without rendering.
        source: Whether to show source as well, and where.
        result: If provided, use as language to format result in a code block.
        tabs: Titles of tabs (if used).
        id: An optional ID for the code block (useful when warning about errors).
        id_prefix: A string used to prefix HTML ids in the generated HTML.
        returncode: The expected exit code.
        transform_source: An optional callable that returns transformed versions of the source.
            The input source is the one that is ran, the output source is the one that is
            rendered (when the source option is enabled).
        session: A session name, to persist state between executed code blocks.
        update_toc: Whether to include generated headings
            into the Markdown table of contents (toc extension).
        **options: Additional options passed from the formatter.

    Returns:
        HTML contents.
    """
    markdown = MarkdownConverter(md, update_toc=update_toc)
    extra = options.get("extra", {})

    if transform_source:
        source_input, source_output = transform_source(code)
    else:
        source_input = code
        source_output = code

    try:
        output = run(source_input, returncode=returncode, session=session, id=id, **extra)
    except ExecutionError as error:
        identifier = id or extra.get("title", "")
        identifier = identifier and f"'{identifier}' "
        exit_message = "errors" if error.returncode is None else f"unexpected code {error.returncode}"
        log_message = (
            f"Execution of {language} code block {identifier}exited with {exit_message}\n\n"
            f"Code block is:\n\n{_format_log_details(source_input)}\n\n"
            f"Output is:\n\n{_format_log_details(str(error), strip_fences=True)}\n"
        )
        logger.warning(log_message)
        return markdown.convert(str(error))

    if html:
        if source:
            placeholder = str(uuid4())
            wrapped_output = add_source(
                source=source_output,
                location=source,
                output=placeholder,
                language=language,
                tabs=tabs,
                **extra,
            )
            return markdown.convert(wrapped_output, stash={placeholder: output})
        return Markup(output)

    wrapped_output = output
    if result and source != "console":
        wrapped_output = code_block(result, output)
    if source:
        wrapped_output = add_source(
            source=source_output,
            location=source,
            output=wrapped_output,
            language=language,
            tabs=tabs,
            result=result,
            **extra,
        )
    prefix = id_prefix if id_prefix is not None else (f"{id}-" if id else None)
    return markdown.convert(wrapped_output, id_prefix=prefix)
