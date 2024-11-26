from __future__ import annotations

import ast
import doctest
import os
import sys
from pathlib import Path

from mypy import api


def validate_results(my_py_results: list[tuple[Path, str, str]]) -> None:
    if len(my_py_results) == 0:
        print("No issues found.")  # noqa: T201
        sys.exit(0)
    else:
        for file, name, result in my_py_results:
            print(f"File: {file}\nFunc or class name: {name}")  # noqa: T201
            print(result)  # noqa: T201
            print()  # noqa: T201
        sys.exit(1)


def check_with_mypy(
    examples: list[tuple[Path, str, str]],
) -> list[tuple[Path, str, str]]:
    results: list[tuple[Path, str, str]] = []
    for file, name, example in examples:
        print(f"Checking {file} {name}")  # noqa: T201
        result = api.run(["-c", example])
        if "Success" in result[0]:
            print("Success")  # noqa: T201
            continue
        print(f"{result[0]}")  # noqa: T201
        results.append((file, name, result[0]))
    return results


def get_docstrings_examples(files: list[Path]) -> list[tuple[Path, str, str]]:
    docstrings: list[tuple[Path, str, str]] = []
    for file in files:
        with open(file) as f:
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                docstring = ast.get_docstring(node)
                if docstring:
                    parsed_examples = doctest.DocTestParser().get_examples(docstring)
                    # Treat all examples as one code block, because there are cases
                    # where explainer text separates code examples
                    example_code = "\n".join(
                        example.source for example in parsed_examples
                    )
                    docstrings.append((file, node.name, example_code))
    return docstrings


def get_python_files() -> list[Path]:
    package_path = Path("./narwhals")
    return [
        Path(path) / name
        for path, _subdir, files in os.walk(package_path)
        for name in files
        if name.endswith(".py")
    ]


if __name__ == "__main__":
    python_files = get_python_files()
    docstring_examples = get_docstrings_examples(python_files)
    my_py_results = check_with_mypy(docstring_examples)
    validate_results(my_py_results)
