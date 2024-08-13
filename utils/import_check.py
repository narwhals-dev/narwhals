from __future__ import annotations

import ast
import sys

BANNED_IMPORTS = {
    "cudf",
    "dask",
    "dask.dataframe",
    "dask_expr",
    "duckdb",
    "ibis",
    "modin",
    "numpy",
    "pandas",
    "polars",
    "pyarrow",
}


class ImportPandasChecker(ast.NodeVisitor):
    def __init__(self, file_name: str, lines: list[str]) -> None:
        self.file_name = file_name
        self.lines = lines
        self.found_import = False

    def visit_If(self, node: ast.If) -> None:  # noqa: N802
        # Check if the condition is `if TYPE_CHECKING`
        if isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING":
            # Skip the body of this if statement
            return
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        for alias in node.names:
            if (
                alias.name in BANNED_IMPORTS
                and "# ignore-banned-import" not in self.lines[node.lineno - 1]
            ):
                print(  # noqa: T201
                    f"{self.file_name}:{node.lineno}:{node.col_offset}: found {alias.name} import"
                )
                self.found_import = True
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        if (
            node.module in BANNED_IMPORTS
            and "# ignore-banned-import" not in self.lines[node.lineno - 1]
        ):
            print(  # noqa: T201
                f"{self.file_name}:{node.lineno}:{node.col_offset}: found {node.module} import"
            )
            self.found_import = True
        self.generic_visit(node)


def check_import_pandas(filename: str) -> bool:
    with open(filename) as file:
        content = file.read()
    tree = ast.parse(content, filename=filename)

    checker = ImportPandasChecker(filename, content.splitlines())
    checker.visit(tree)

    return checker.found_import


if __name__ == "__main__":
    ret = 0
    for filename in sys.argv[1:]:
        if not filename.endswith(".py"):
            continue
        ret |= check_import_pandas(filename)
    sys.exit(ret)
