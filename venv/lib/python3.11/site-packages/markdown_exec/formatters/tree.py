"""Formatter for file-system trees."""

from __future__ import annotations

from textwrap import dedent
from typing import TYPE_CHECKING, Any

from markdown_exec.rendering import MarkdownConverter, code_block

if TYPE_CHECKING:
    from markdown import Markdown


def _rec_build_tree(lines: list[str], parent: list, offset: int, base_indent: int) -> int:
    while offset < len(lines):
        line = lines[offset]
        lstripped = line.lstrip()
        indent = len(line) - len(lstripped)
        if indent == base_indent:
            parent.append((lstripped, []))
            offset += 1
        elif indent > base_indent:
            offset = _rec_build_tree(lines, parent[-1][1], offset, indent)
        else:
            return offset
    return offset


def _build_tree(code: str) -> list[tuple[str, list]]:
    lines = dedent(code.strip()).split("\n")
    root_layer: list[tuple[str, list]] = []
    _rec_build_tree(lines, root_layer, 0, 0)
    return root_layer


def _rec_format_tree(tree: list[tuple[str, list]], *, root: bool = True) -> list[str]:
    lines = []
    n_items = len(tree)
    for index, node in enumerate(tree):
        last = index == n_items - 1
        prefix = "" if root else f"{'└' if last else '├'}── "
        if node[1]:
            lines.append(f"{prefix}📁 {node[0]}")
            sublines = _rec_format_tree(node[1], root=False)
            if root:
                lines.extend(sublines)
            else:
                indent_char = " " if last else "│"
                lines.extend([f"{indent_char}   {line}" for line in sublines])
        else:
            name = node[0].split()[0]
            icon = "📁" if name.endswith("/") else "📄"
            lines.append(f"{prefix}{icon} {node[0]}")
    return lines


def _format_tree(code: str, md: Markdown, result: str, **options: Any) -> str:
    markdown = MarkdownConverter(md)
    output = "\n".join(_rec_format_tree(_build_tree(code)))
    return markdown.convert(code_block(result or "bash", output, **options.get("extra", {})))
