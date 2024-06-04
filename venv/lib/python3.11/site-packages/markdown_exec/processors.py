"""This module contains a Markdown extension allowing to integrate generated headings into the ToC."""

from __future__ import annotations

import copy
import re
from typing import TYPE_CHECKING
from xml.etree.ElementTree import Element

from markdown.treeprocessors import Treeprocessor
from markdown.util import HTML_PLACEHOLDER_RE

if TYPE_CHECKING:
    from markdown import Markdown
    from markupsafe import Markup


# code taken from mkdocstrings, credits to @oprypin
class IdPrependingTreeprocessor(Treeprocessor):
    """Prepend the configured prefix to IDs of all HTML elements."""

    name = "markdown_exec_ids"

    def __init__(self, md: Markdown, id_prefix: str) -> None:  # noqa: D107
        super().__init__(md)
        self.id_prefix = id_prefix

    def run(self, root: Element) -> None:  # noqa: D102
        if not self.id_prefix:
            return
        for el in root.iter():
            id_attr = el.get("id")
            if id_attr:
                el.set("id", self.id_prefix + id_attr)

            href_attr = el.get("href")
            if href_attr and href_attr.startswith("#"):
                el.set("href", "#" + self.id_prefix + href_attr[1:])

            name_attr = el.get("name")
            if name_attr:
                el.set("name", self.id_prefix + name_attr)

            if el.tag == "label":
                for_attr = el.get("for")
                if for_attr:
                    el.set("for", self.id_prefix + for_attr)


# code taken from mkdocstrings, credits to @oprypin
class HeadingReportingTreeprocessor(Treeprocessor):
    """Records the heading elements encountered in the document."""

    name = "markdown_exec_record_headings"
    regex = re.compile("[Hh][1-6]")

    def __init__(self, md: Markdown, headings: list[Element]):  # noqa: D107
        super().__init__(md)
        self.headings = headings

    def run(self, root: Element) -> None:  # noqa: D102
        for el in root.iter():
            if self.regex.fullmatch(el.tag):
                el = copy.copy(el)  # noqa: PLW2901
                # 'toc' extension's first pass (which we require to build heading stubs/ids) also edits the HTML.
                # Undo the permalink edit so we can pass this heading to the outer pass of the 'toc' extension.
                if len(el) > 0 and el[-1].get("class") == self.md.treeprocessors["toc"].permalink_class:  # type: ignore[attr-defined]
                    del el[-1]
                self.headings.append(el)


class InsertHeadings(Treeprocessor):
    """Our headings insertor."""

    name = "markdown_exec_insert_headings"

    def __init__(self, md: Markdown):
        """Initialize the object.

        Arguments:
            md: A `markdown.Markdown` instance.
        """
        super().__init__(md)
        self.headings: dict[Markup, list[Element]] = {}

    def run(self, root: Element) -> None:  # noqa: D102 (ignore missing docstring)
        if not self.headings:
            return

        for el in root.iter():
            match = HTML_PLACEHOLDER_RE.match(el.text or "")
            if match:
                counter = int(match.group(1))
                markup: Markup = self.md.htmlStash.rawHtmlBlocks[counter]  # type: ignore[assignment]
                if markup in self.headings:
                    div = Element("div", {"class": "markdown-exec"})
                    div.extend(self.headings[markup])
                    el.append(div)


class RemoveHeadings(Treeprocessor):
    """Our headings remover."""

    name = "markdown_exec_remove_headings"

    def run(self, root: Element) -> None:  # noqa: D102
        carry_text = ""
        for el in reversed(root):  # Reversed mainly for the ability to mutate during iteration.
            for subel in reversed(el):
                if subel.tag == "div" and subel.get("class") == "markdown-exec":
                    # Delete the duplicated headings along with their container, but keep the text (i.e. the actual HTML).
                    carry_text = (subel.text or "") + carry_text
                    el.remove(subel)
                elif carry_text:
                    subel.tail = (subel.tail or "") + carry_text
                    carry_text = ""
            if carry_text:
                el.text = (el.text or "") + carry_text
