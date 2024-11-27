"""Run ruff on Python examples in docstrings."""

from __future__ import annotations

import ast
import doctest
import subprocess
import sys
import tempfile
from pathlib import Path


def extract_docstring_examples(files: list[Path]) -> list[tuple[Path, str, str]]:
    """Extract examples from docstrings in Python files."""
    examples: list[tuple[Path, str, str]] = []

    for file in files:
        with open(file, encoding="utf-8") as f:
            tree = ast.parse(f.read())

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                docstring = ast.get_docstring(node)
                if docstring:
                    parsed_examples = doctest.DocTestParser().get_examples(docstring)
                    example_code = "\n".join(
                        example.source for example in parsed_examples
                    )
                    if example_code.strip():
                        examples.append((file, node.name, example_code))

    return examples


def create_temp_files(examples: list[tuple[Path, str, str]]) -> list[tuple[Path, str]]:
    """Create temporary files for all examples and return their paths."""
    temp_files: list[tuple[Path, str]] = []

    for file, name, example in examples:
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)  # noqa: SIM115
        temp_file.write(example)
        temp_file_path = temp_file.name
        temp_file.close()
        temp_files.append((Path(temp_file_path), f"{file}:{name}"))

    return temp_files


def run_ruff_on_temp_files(temp_files: list[tuple[Path, str]]) -> list[str]:
    """Run ruff on all temporary files and collect error messages."""
    temp_file_paths = [str(temp_file[0]) for temp_file in temp_files]

    result = subprocess.run(  # noqa: S603
        [  # noqa: S607
            "python",
            "-m",
            "ruff",
            "check",
            "--select=F",
            "--ignore=F811",
            *temp_file_paths,
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode == 0:
        return []  # No issues found
    return result.stdout.splitlines()  # Return ruff errors as a list of lines


def report_errors(errors: list[str], temp_files: list[tuple[Path, str]]) -> None:
    """Map errors back to original examples and report them."""
    if not errors:
        return

    print("❌ Ruff issues found in examples:\n")  # noqa: T201
    for line in errors:
        for temp_file, original_context in temp_files:
            if str(temp_file) in line:
                print(f"{original_context}{line.replace(str(temp_file), '')}")  # noqa: T201
                break


def cleanup_temp_files(temp_files: list[tuple[Path, str]]) -> None:
    """Remove all temporary files."""
    for temp_file, _ in temp_files:
        temp_file.unlink()


def main(python_files: list[str]) -> None:
    docstring_examples = extract_docstring_examples(python_files)

    if not docstring_examples:
        sys.exit(0)

    temp_files = create_temp_files(docstring_examples)

    try:
        errors = run_ruff_on_temp_files(temp_files)
        report_errors(errors, temp_files)
    finally:
        cleanup_temp_files(temp_files)

    if errors:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main(sys.argv[1:])
