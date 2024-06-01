"""Markdown extensions and helpers."""

from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
from textwrap import indent
from typing import TYPE_CHECKING, Any, Iterator

from markdown import Markdown
from markupsafe import Markup

from markdown_exec.processors import (
    HeadingReportingTreeprocessor,
    IdPrependingTreeprocessor,
    InsertHeadings,
    RemoveHeadings,
)

if TYPE_CHECKING:
    from xml.etree.ElementTree import Element

    from markdown import Extension


def code_block(language: str, code: str, **options: str) -> str:
    """Format code as a code block.

    Parameters:
        language: The code block language.
        code: The source code to format.
        **options: Additional options passed from the source, to add back to the generated code block.

    Returns:
        The formatted code block.
    """
    opts = " ".join(f'{opt_name}="{opt_value}"' for opt_name, opt_value in options.items())
    return f"````````{language} {opts}\n{code}\n````````"


def tabbed(*tabs: tuple[str, str]) -> str:
    """Format tabs using `pymdownx.tabbed` extension.

    Parameters:
        *tabs: Tuples of strings: title and text.

    Returns:
        The formatted tabs.
    """
    parts = []
    for title, text in tabs:
        title = title.replace(r"\|", "|").strip()  # noqa: PLW2901
        parts.append(f'=== "{title}"')
        parts.append(indent(text, prefix=" " * 4))
        parts.append("")
    return "\n".join(parts)


def _hide_lines(source: str) -> str:
    return "\n".join(line for line in source.split("\n") if "markdown-exec: hide" not in line).strip()


def add_source(
    *,
    source: str,
    location: str,
    output: str,
    language: str,
    tabs: tuple[str, str],
    result: str = "",
    **extra: str,
) -> str:
    """Add source code block to the output.

    Parameters:
        source: The source code block.
        location: Where to add the source (above, below, tabbed-left, tabbed-right, console).
        output: The current output.
        language: The code language.
        tabs: Tabs titles (if used).
        result: Syntax to use when concatenating source and result with "console" location.
        **extra: Extra options added back to source code block.

    Raises:
        ValueError: When the given location is not supported.

    Returns:
        The updated output.
    """
    source = _hide_lines(source)
    if location == "console":
        return code_block(result or language, source + "\n" + output, **extra)

    source_block = code_block(language, source, **extra)
    if location == "above":
        return source_block + "\n\n" + output
    if location == "below":
        return output + "\n\n" + source_block
    if location == "material-block":
        return source_block + f'\n\n<div class="result" markdown="1" >\n\n{output}\n\n</div>'

    source_tab_title, result_tab_title = tabs
    if location == "tabbed-left":
        return tabbed((source_tab_title, source_block), (result_tab_title, output))
    if location == "tabbed-right":
        return tabbed((result_tab_title, output), (source_tab_title, source_block))

    raise ValueError(f"unsupported location for sources: {location}")


class MarkdownConfig:
    """This class returns a singleton used to store Markdown extensions configuration.

    You don't have to instantiate the singleton yourself:
    we provide it as [`markdown_config`][markdown_exec.rendering.markdown_config].
    """

    _singleton: MarkdownConfig | None = None

    def __new__(cls) -> MarkdownConfig:  # noqa: D102,PYI034
        if cls._singleton is None:
            cls._singleton = super().__new__(cls)
        return cls._singleton

    def __init__(self) -> None:  # noqa: D107
        self.exts: list[str] | None = None
        self.exts_config: dict[str, dict[str, Any]] | None = None

    def save(self, exts: list[str], exts_config: dict[str, dict[str, Any]]) -> None:
        """Save Markdown extensions and their configuration.

        Parameters:
            exts: The Markdown extensions.
            exts_config: The extensions configuration.
        """
        self.exts = exts
        self.exts_config = exts_config

    def reset(self) -> None:
        """Reset Markdown extensions and their configuration."""
        self.exts = None
        self.exts_config = None


markdown_config = MarkdownConfig()
"""This object can be used to save the configuration of your Markdown extensions.

For example, since we provide a MkDocs plugin, we use it to store the configuration
that was read from `mkdocs.yml`:

```python
from markdown_exec.rendering import markdown_config

# ...in relevant events/hooks, access and modify extensions and their configs, then:
markdown_config.save(extensions, extensions_config)
```

See the actual event hook: [`on_config`][markdown_exec.mkdocs_plugin.MarkdownExecPlugin.on_config].
See the [`save`][markdown_exec.rendering.MarkdownConfig.save]
and [`reset`][markdown_exec.rendering.MarkdownConfig.reset] methods.

Without it, Markdown Exec will rely on the `registeredExtensions` attribute
of the original Markdown instance, which does not forward everything
that was configured, notably extensions like `tables`. Other extensions
such as `attr_list` are visible, but fail to register properly when
reusing their instances. It means that the rendered HTML might differ
from what you expect (tables not rendered, attribute lists not injected,
emojis not working, etc.).
"""

# FIXME: When a heading contains an XML entity such as &mdash;,
# the entity is stashed and replaced with a placeholder.
# The heading therefore contains this placeholder.
# When reporting the heading to the upper conversion layer (for the ToC),
# the placeholder gets unstashed using the upper Markdown instance
# instead of the neste one. If the upper instance doesn't know the placeholder,
# nothing happens. But if it knows it, we then get a heading with garbabe/previous
# contents within it, messing up the ToC.
# We should fix this somehow. In the meantime, the workaround is to avoid
# XML entities that get stashed in headings.


@lru_cache(maxsize=None)
def _register_headings_processors(md: Markdown) -> None:
    md.treeprocessors.register(
        InsertHeadings(md),
        InsertHeadings.name,
        priority=75,  # right before markdown.blockprocessors.HashHeaderProcessor
    )
    md.treeprocessors.register(
        RemoveHeadings(md),
        RemoveHeadings.name,
        priority=4,  # right after toc
    )


def _mimic(md: Markdown, headings: list[Element], *, update_toc: bool = True) -> Markdown:
    new_md = Markdown()
    extensions: list[Extension | str] = markdown_config.exts or md.registeredExtensions  # type: ignore[assignment]
    extensions_config: dict[str, dict[str, Any]] = markdown_config.exts_config or {}
    new_md.registerExtensions(extensions, extensions_config)
    new_md.treeprocessors.register(
        IdPrependingTreeprocessor(md, ""),
        IdPrependingTreeprocessor.name,
        priority=4,  # right after 'toc' (needed because that extension adds ids to headings)
    )
    new_md._original_md = md  # type: ignore[attr-defined]

    if update_toc:
        _register_headings_processors(md)
        new_md.treeprocessors.register(
            HeadingReportingTreeprocessor(new_md, headings),
            HeadingReportingTreeprocessor.name,
            priority=1,  # Close to the end.
        )

    return new_md


@contextmanager
def _id_prefix(md: Markdown, prefix: str | None) -> Iterator[None]:
    MarkdownConverter.counter += 1
    id_prepending_processor = md.treeprocessors[IdPrependingTreeprocessor.name]
    id_prepending_processor.id_prefix = prefix if prefix is not None else f"exec-{MarkdownConverter.counter}--"  # type: ignore[attr-defined]
    try:
        yield
    finally:
        id_prepending_processor.id_prefix = ""  # type: ignore[attr-defined]


class MarkdownConverter:
    """Helper class to avoid breaking the original Markdown instance state."""

    counter: int = 0

    def __init__(self, md: Markdown, *, update_toc: bool = True) -> None:  # noqa: D107
        self._md_ref: Markdown = md
        self._headings: list[Element] = []
        self._update_toc = update_toc

    @property
    def _original_md(self) -> Markdown:
        return getattr(self._md_ref, "_original_md", self._md_ref)

    def _report_headings(self, markup: Markup) -> None:
        self._original_md.treeprocessors[InsertHeadings.name].headings[markup] = self._headings  # type: ignore[attr-defined]
        self._headings = []

    def convert(self, text: str, stash: dict[str, str] | None = None, id_prefix: str | None = None) -> Markup:
        """Convert Markdown text to safe HTML.

        Parameters:
            text: Markdown text.
            stash: An HTML stash.

        Returns:
            Safe HTML.
        """
        md = _mimic(self._original_md, self._headings, update_toc=self._update_toc)

        # convert markdown to html
        with _id_prefix(md, id_prefix):
            converted = md.convert(text)

        # restore html from stash
        for placeholder, stashed in (stash or {}).items():
            converted = converted.replace(placeholder, stashed)

        markup = Markup(converted)

        # pass headings to upstream conversion layer
        if self._update_toc:
            self._report_headings(markup)

        return markup
