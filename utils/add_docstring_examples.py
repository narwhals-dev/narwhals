"""Add docstring examples to docstrings.

In order to keep Narwhals lightweight, we keep lengthy docstring examples
in `docs/docstring_examples`. These then get dynamically added to the
docstrings in CI before running doctests and before publishing docs.

To run it locally and add docstring examples to all tracked files, you
can run (on mac/linux):

    git ls-files narwhals | xargs python utils/add_docstring_examples.py
"""

from __future__ import annotations

import ast
import importlib
import sys
from ast import NodeVisitor


def visit_node(
    node: ast.FunctionDef | ast.ClassDef, examples: dict[str, str, int]
) -> tuple[str, bool] | None:
    """Visit node.

    Returns:
        - If the node has a docstring, and there is a docstring example stored for
          this function, then return a tuple with

          - new docstring example
          - a boolean indicating whether the function already had a docstring to begin with
          - the current docstring lineno.
        - Else, return None.
    """
    if (
        node.name in examples
        and node.body
        and isinstance(expr := node.body[0], ast.Expr)
        and isinstance(value := expr.value, ast.Constant)
        and isinstance(docstring := value.value, str)
        and value.end_lineno is not None
    ):
        # Subtract 1 as end_lineno is 1-indexed.
        return examples[node.name], "Examples:" in docstring, value.end_lineno - 1
    return None


class Visitor(NodeVisitor):
    def __init__(self, file: str) -> None:
        self.file = file
        self.additions: dict[int, str] = {}
        self.already_has_docstring_example: dict[int, bool] = {}
        self.examples: dict[str, str] = importlib.import_module(
            self.file.replace("narwhals", "docs.docstring_examples")
            .replace("/", ".")
            .removesuffix(".py")
        ).EXAMPLES

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        if (result := visit_node(node, self.examples)) is not None:
            new_docstring_example, docstring_already_has_example, lineno = result
            self.additions[lineno] = new_docstring_example
            self.already_has_docstring_example[lineno] = docstring_already_has_example
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        if (result := visit_node(node, self.examples)) is not None:
            new_docstring_example, docstring_already_has_example, lineno = result
            self.additions[lineno] = new_docstring_example
            self.already_has_docstring_example[lineno] = docstring_already_has_example
        self.generic_visit(node)


if __name__ == "__main__":
    files = sys.argv[1:]
    for file in files:
        if not file.endswith(".py"):
            # Skip non-Python files.
            continue
        with open(file) as fd:
            content = fd.read()
        tree = ast.parse(content)
        try:
            visitor = Visitor(file)
        except (AttributeError, ModuleNotFoundError):
            # There are no docstrings examples to replace
            # for this file.
            continue
        visitor.visit(tree)
        if visitor.additions:
            lines = content.splitlines()
            for lineno, addition in visitor.additions.items():
                line = lines[lineno]
                indent = len(line) - len(line.lstrip())
                rewritten_line = line.rstrip().removesuffix('"""') + "\n"
                if not visitor.already_has_docstring_example[lineno]:
                    rewritten_line += " " * indent + "Examples:\n"
                rewritten_line += addition.lstrip("\n")
                rewritten_line += '"""\n'
                lines[lineno] = rewritten_line
            with open(file, "w") as fd:
                fd.write("\n".join(lines) + "\n")
