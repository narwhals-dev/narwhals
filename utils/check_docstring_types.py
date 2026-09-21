"""Type-check the Python examples embedded in docstrings.

No type checker reads docstrings, so we hand them the examples ourselves: each docstring
becomes one throwaway module in a temporary directory, and every checker is then pointed
at that directory. Blank lines pad each generated module so that every statement keeps
the line number it has in the original source file, and diagnostics therefore point at
a clickable location in the real file.

Usage:
    python utils/check_docstring_types.py [--checker mypy] [PATH ...]
"""

from __future__ import annotations

import argparse
import ast
import doctest
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_PATHS = (REPO_ROOT / "src" / "narwhals",)

# Examples are illustrative snippets, not library code:
# - `arg-type`: mypy joins a heterogeneous dict literal to `dict[str, object]`, which no
#   `Mapping[str, <specific>]` parameter accepts. Pyright and pyrefly infer a union
#   instead, so genuine argument mismatches are still caught, by them.
# - `no-untyped-def`, `no-untyped-call`: some examples deliberately show untyped user
#   code (see `narwhalify`).
# - `no-redef`: a docstring's examples share one namespace and may rebind a name to
#   contrast two ways of writing the same thing. `check_docstrings.py` ignores `F811`
#   for the same reason.
MYPY_DISABLED_CODES = ("arg-type", "no-redef", "no-untyped-call", "no-untyped-def")

# Pyright's equivalent of `no-redef`, on line 1 so it precedes every example.
PYRIGHT_PRAGMA = "# pyright: reportRedeclaration=false"

CHECKERS: dict[str, tuple[str, ...]] = {
    "mypy": (
        "mypy",
        "--no-incremental",
        *(f"--disable-error-code={code}" for code in MYPY_DISABLED_CODES),
    ),
    "pyright": ("pyright",),
    # Unlike the other two, pyrefly resolves its config from the files being checked, not
    # from the working directory, so it needs to be pointed at ours explicitly.
    "pyrefly": ("pyrefly", "check", "-c", "pyproject.toml"),
}


def iter_python_files(paths: Sequence[Path]) -> Iterator[Path]:
    """Yield the files to extract from, skipping private modules when walking a directory.

    `doctest` runs examples with the defining module's globals, which we cannot replicate
    here, so examples in private modules lean on names they never import. Public
    docstrings are self-contained, and a private module can still be passed explicitly.
    """
    for path in (p.resolve() for p in paths):
        if not path.is_dir():
            yield path
            continue
        for file in sorted(path.rglob("*.py")):
            parts = file.relative_to(path).parts
            if not any(p.startswith("_") and p != "__init__.py" for p in parts):
                yield file


def iter_docstrings(module: ast.Module) -> Iterator[tuple[str, int]]:
    """Yield every docstring in a parsed module, with the line its literal starts on.

    Any bare string statement is a docstring: `ast.get_docstring` would see only the
    module, class and function ones, and miss those documenting an attribute.
    """
    for node in ast.walk(module):
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(text := node.value.value, str)
        ):
            yield text, node.value.lineno


def render_examples(docstring: str, lineno: int) -> str:
    """Render the examples of a docstring starting on `lineno` as aligned module source.

    Returns an empty string when the docstring has no examples.
    """
    lines = [PYRIGHT_PRAGMA]
    for example in doctest.DocTestParser().get_examples(docstring):
        lines.extend([""] * (lineno + example.lineno - len(lines) - 1))
        lines.extend(example.source.splitlines())
    return "\n".join(lines) + "\n" if len(lines) > 1 else ""


def verify_alignment(
    module_source: str, original_lines: Sequence[str], file: Path
) -> None:
    """Raise unless every rendered statement sits on top of the line it came from.

    Drift would go unnoticed and report every diagnostic against the wrong line, so a
    crash is the better outcome. `original_lines` are the lines of `file`, which is only
    named in the error message.
    """
    for i, rendered in enumerate(module_source.splitlines()):
        if i == 0 or not rendered:  # Line 1 is the pragma, blank lines are the padding.
            continue
        original = original_lines[i] if i < len(original_lines) else ""
        # Drop the `>>> ` or `... ` prompt to recover what was rendered from it.
        if original.lstrip()[4:] != rendered:
            msg = (
                f"{file}:{i + 1} holds {original!r}, but a docstring example was "
                f"rendered onto it as {rendered!r}. Line alignment is broken."
            )
            raise AssertionError(msg)


def write_modules(files: Iterable[Path], work_dir: Path) -> dict[Path, Path]:
    """Write one module per docstring into `work_dir`, mapped to the file it came from."""
    sources: dict[Path, Path] = {}
    for file in files:
        text = file.read_text("utf-8")
        original_lines = text.splitlines()
        for docstring, lineno in iter_docstrings(ast.parse(text)):
            if not (module_source := render_examples(docstring, lineno)):
                continue
            verify_alignment(module_source, original_lines, file)
            path = work_dir / f"d{len(sources):04d}_{file.stem}.py"
            path.write_text(module_source, "utf-8")
            sources[path] = file.relative_to(REPO_ROOT)
    return sources


def run_checker(checker: str, work_dir: Path, sources: Mapping[Path, Path]) -> int:
    """Run one type checker over every generated module, reporting against real paths."""
    print(f"\n>>> {' '.join(CHECKERS[checker])}", flush=True)
    result = subprocess.run(  # noqa: S603
        [*CHECKERS[checker], str(work_dir)],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO_ROOT,
    )
    output = result.stdout + result.stderr
    for generated, source in sources.items():
        output = output.replace(str(generated), str(source))
    print(output, flush=True)
    return result.returncode


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, default=DEFAULT_PATHS)
    parser.add_argument("--checker", action="append", choices=sorted(CHECKERS))
    args = parser.parse_args(argv)

    with tempfile.TemporaryDirectory() as tmp:
        work_dir = Path(tmp)
        sources = write_modules(iter_python_files(args.paths), work_dir)
        # An empty run is a broken one, and mypy already fails on an empty directory.
        print(f"Found examples in {len(sources)} docstrings.")
        codes = [
            run_checker(checker, work_dir, sources)
            for checker in args.checker or CHECKERS
        ]
    # Not `max`: a checker killed by a signal reports a negative return code.
    return 0 if all(code == 0 for code in codes) else 1


if __name__ == "__main__":
    sys.exit(main())
