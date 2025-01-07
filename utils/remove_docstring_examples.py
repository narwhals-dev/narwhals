from __future__ import annotations

import ast
import sys
from ast import NodeVisitor


class Visitor(NodeVisitor):
    def __init__(self, file: str) -> None:
        self.file = file
        self.to_remove: list[tuple[int, int]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        if (
            node.body
            and isinstance(expr := node.body[0], ast.Expr)
            and isinstance(value := expr.value, ast.Constant)
            and isinstance(docstring := value.value, str)
            and "Examples:" in docstring
            and value.end_lineno is not None
        ):
            examples_line_start = (
                value.lineno
                + [line.strip() for line in docstring.splitlines()].index("Examples:")
                - 1
            )
            examples_line_end = value.end_lineno - 1
            self.to_remove.append((examples_line_start, examples_line_end))

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
            for removal in sorted(visitor.to_remove, key=lambda x: -x[0]):
                del lines[removal[0] : removal[1]]
            with open(file, "w") as fd:
                fd.write("\n".join(lines))
