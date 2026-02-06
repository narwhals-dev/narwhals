"""Run ruff on Python examples in docstrings."""

from __future__ import annotations

import ast
import doctest
import os
import subprocess as sp
import sys
import sysconfig
import tempfile
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from typing_extensions import TypeAlias

ExitCode: TypeAlias = "Literal[0, 1] | None"
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


def ruff_check(*paths: Path, select: Iterable[str], ignore: Iterable[str]) -> Literal[0]:
    ruff = find_ruff_bin()
    select = f"--select={','.join(select)}"
    ignore = f"--ignore={','.join(ignore)}"
    args = (ruff, "check", select, ignore, *paths)
    sp.run(args, capture_output=True, text=True, check=True)
    return 0


parser = doctest.DocTestParser()


def _try_parse_examples(node: ast.AST) -> tuple[NodeName, Code] | None:
    if (
        isinstance(node, (ast.FunctionDef, ast.ClassDef))
        and (doc := ast.get_docstring(node))
        and (code := "\n".join(e.source.strip() for e in parser.get_examples(doc)))
    ):
        return node.name, code
    return None


def iter_docstring_examples(files: Iterable[str | Path]) -> Iterator[DocstringExample]:
    """Extract examples from docstrings in Python files."""
    for file in files:
        fp = Path(file)
        for node in ast.walk(ast.parse(fp.read_bytes())):
            if example := _try_parse_examples(node):
                yield (fp, *example)


def _restore_context(
    error: sp.CalledProcessError, files: Iterable[tuple[Path, OriginalContext]]
) -> str:
    output = str(error.output)
    for file, context in files:
        output = output.replace(str(file), context)
    return output


def main(python_files: Iterable[str | Path]) -> ExitCode:
    with tempfile.TemporaryDirectory() as tmp:
        temp_dir = Path(tmp)
        temp_files: deque[tuple[Path, OriginalContext]] = deque()
        for i, (file, name, example) in enumerate(iter_docstring_examples(python_files)):
            # TODO @dangotbanned: Could this be kept lazy?
            # Iterator should yield these bits instead
            temp_path = temp_dir / f"{name}_{i}.py"
            temp_path.write_text(example)
            temp_files.append((temp_path, f"{file.as_posix()}:{name}"))
        try:
            return ruff_check(*temp_dir.iterdir(), select=SELECT, ignore=IGNORE)
        except sp.CalledProcessError as err:
            msg = "Ruff issues found in examples"
            print(f"{msg}:\n\n{_restore_context(err, temp_files)}")
            return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
