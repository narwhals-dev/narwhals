from __future__ import annotations

import ast
import sys
from ast import NodeVisitor


def find_docstring_example(
    node: ast.FunctionDef | ast.ClassDef,
) -> tuple[int, int] | None:
    """If node contains a docstring example, return start and end lines."""
    if (
        node.body
        and isinstance(expr := node.body[0], ast.Expr)
        and isinstance(value := expr.value, ast.Constant)
        and isinstance(docstring := value.value, str)
        and "Examples:" in docstring
        and value.end_lineno is not None
    ):
        docstring_lines = [line.strip() for line in docstring.splitlines()]
        examples_line_start = value.lineno + docstring_lines.index("Examples:")
        # lineno is 1-indexed so we subtract 1.
        return (examples_line_start - 1, value.end_lineno - 1)
    return None


class Visitor(NodeVisitor):
    def __init__(self, file: str) -> None:
        self.file = file
        self.to_remove: list[tuple[int, int]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        if removal := find_docstring_example(node):
            self.to_remove.append(removal)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        if removal := find_docstring_example(node):
            self.to_remove.append(removal)
        self.generic_visit(node)


if __name__ == "__main__":
    files = sys.argv[1:]
    for file in files:
        if not file.endswith(".py"):
            continue
        with open(file) as fd:
            content = fd.read()
        tree = ast.parse(content)
        visitor = Visitor(file)
        visitor.visit(tree)
        if visitor.to_remove:
            lines = content.splitlines()
            removals = sorted(visitor.to_remove, key=lambda x: -x[0])
            for examples_start, examples_end in removals:
                del lines[examples_start:examples_end]
            with open(file, "w") as fd:
                fd.write("\n".join(lines) + "\n")
