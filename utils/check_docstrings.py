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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from typing_extensions import TypeAlias

NodeName: TypeAlias = str
Code: TypeAlias = str
DocstringExample: TypeAlias = tuple[Path, NodeName, Code]
OriginalContext: TypeAlias = str

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


parser = doctest.DocTestParser()


def try_parse(node: ast.AST) -> tuple[NodeName, Code] | None:
    if (
        isinstance(node, (ast.FunctionDef, ast.ClassDef))
        and (doc := ast.get_docstring(node))
        and (code := "\n".join(e.source for e in parser.get_examples(doc)).strip())
    ):
        return node.name, code
    return None


def iter_docstring_examples(files: Iterable[str | Path]) -> Iterator[DocstringExample]:
    """Extract examples from docstrings in Python files."""
    for file in files:
        fp = Path(file)
        for node in ast.walk(ast.parse(fp.read_text("utf-8"))):
            if example := try_parse(node):
                yield (fp, *example)


def main(python_files: list[str]) -> None:
    # TODO @dangotbanned: Could this be kept lazy?
    it = iter_docstring_examples(python_files)
    if docstring_examples := tuple(it):
        with tempfile.TemporaryDirectory() as tmpdir:
            # `create_temp_files`
            # Create temporary files for all examples and return their paths.
            tmpdir_path = Path(tmpdir)
            temp_files: list[tuple[Path, OriginalContext]] = []
            for i, (file, name, example) in enumerate(docstring_examples):
                temp_path = tmpdir_path / f"{name}_{i}.py"
                temp_path.write_text(example)
                temp_files.append((temp_path, f"{file.as_posix()}:{name}"))

            # `run_ruff_on_temp_files`
            # Run ruff on all temporary files and collect error messages.
            select = f"--select={','.join(SELECT)}"
            ignore = f"--ignore={','.join(IGNORE)}"
            args = [find_ruff_bin(), "check", select, ignore, *tmpdir_path.iterdir()]
            result = subprocess.run(args, capture_output=True, text=True, check=False)  # noqa: S603
            if result.returncode:
                # `report_errors`
                # Map errors back to original examples and report them
                print("Ruff issues found in examples:\n")
                stdout = result.stdout
                for temp_file, original_context in temp_files:
                    stdout = stdout.replace(str(temp_file), original_context)
                print(stdout)
                sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main(sys.argv[1:])
