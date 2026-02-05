"""Run ruff on Python examples in docstrings."""

from __future__ import annotations

import ast
import doctest
import os
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path
from subprocess import CompletedProcess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

DocstringExamples: TypeAlias = list[tuple[Path, str, str]]
TempFiles: TypeAlias = list[tuple[Path, str]]

SELECT = (
    "F",  # pyflakes-f
)
IGNORE = (
    "F811",  # redefined-while-unused
    "F821",  # undefined-name (misses https://docs.python.org/3/library/doctest.html#what-s-the-execution-context)
)


def find_ruff_bin() -> Path:
    """Return the ruff binary path.

    Adapted from [`ruff.__main__.find_ruff_bin`], see also [astral-sh/ruff#18153], [astral-sh/uv#1677].

    [`ruff.__main__.find_ruff_bin`]: https://github.com/astral-sh/ruff/blob/2d6ca092fa1655f14f10dab6e2a5b95f5f682c24/python/ruff/__main__.py
    [astral-sh/ruff#18153]: https://github.com/astral-sh/ruff/issues/18153#issuecomment-2888581114
    [astral-sh/uv#1677]: https://github.com/astral-sh/uv/issues/1677
    """
    ruff_exe: str = "ruff" + sysconfig.get_config_var("EXE")

    scripts_path = Path(sysconfig.get_path("scripts")) / ruff_exe
    if scripts_path.is_file():
        return scripts_path

    if sys.version_info >= (3, 10):
        user_scheme = sysconfig.get_preferred_scheme("user")
    elif os.name == "nt":
        user_scheme = "nt_user"
    elif sys.platform == "darwin" and sys._framework:
        user_scheme = "osx_framework_user"
    else:
        user_scheme = "posix_user"

    user_path = Path(sysconfig.get_path("scripts", scheme=user_scheme)) / ruff_exe
    if user_path.is_file():
        return user_path
    msg = (
        f"Unable to find ruff at:\n- {scripts_path.as_posix()}\n- {user_path.as_posix()}\n\n"
        "Hint: did you follow this guide? https://github.com/narwhals-dev/narwhals?tab=contributing-ov-file#readme"
    )
    raise FileNotFoundError(msg)


def extract_docstring_examples(files: list[str]) -> DocstringExamples:
    """Extract examples from docstrings in Python files."""
    examples: DocstringExamples = []

    for file in files:  # noqa: PLR1702
        fp = Path(file)
        tree = ast.parse(fp.read_text("utf-8"))

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                docstring = ast.get_docstring(node)
                if docstring:
                    parsed_examples = doctest.DocTestParser().get_examples(docstring)
                    example_code = "\n".join(
                        example.source for example in parsed_examples
                    )
                    if example_code.strip():
                        examples.append((fp, node.name, example_code))

    return examples


def create_temp_files(examples: DocstringExamples) -> TempFiles:
    """Create temporary files for all examples and return their paths."""
    temp_files: TempFiles = []

    for file, name, example in examples:
        temp_file = tempfile.NamedTemporaryFile(  # noqa: SIM115
            encoding="utf-8", mode="w", suffix=".py", delete=False
        )
        temp_file.write(example)
        temp_file_path = temp_file.name
        temp_file.close()
        temp_files.append((Path(temp_file_path), f"{file!s}:{name}"))

    return temp_files


def run_ruff_on_temp_files(temp_files: TempFiles) -> CompletedProcess[str] | None:
    """Run ruff on all temporary files and collect error messages."""
    temp_file_paths = [temp_file[0] for temp_file in temp_files]
    select = f"--select={','.join(SELECT)}"
    ignore = f"--ignore={','.join(IGNORE)}"
    result = subprocess.run(  # noqa: S603
        [find_ruff_bin(), "check", select, ignore, *temp_file_paths],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode == 0:
        return None
    return result


def report_errors(completed: CompletedProcess[str], temp_files: TempFiles) -> None:
    """Map errors back to original examples and report them."""
    print("Ruff issues found in examples:\n")
    stdout = completed.stdout
    for temp_file, original_context in temp_files:
        stdout = stdout.replace(str(temp_file), original_context)
    print(stdout)


def cleanup_temp_files(temp_files: TempFiles) -> None:
    """Remove all temporary files."""
    for temp_file, _ in temp_files:
        temp_file.unlink()


def main(python_files: list[str]) -> None:
    if docstring_examples := extract_docstring_examples(python_files):
        temp_files = create_temp_files(docstring_examples)
        try:
            if errors := run_ruff_on_temp_files(temp_files):
                report_errors(errors, temp_files)
                sys.exit(1)
        finally:
            cleanup_temp_files(temp_files)
    sys.exit(0)


if __name__ == "__main__":
    main(sys.argv[1:])
