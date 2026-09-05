#!/usr/bin/env python3
"""Run narwhals' own test suite against `narwhals-dict`.

Mirrors the approach of `narwhals-daft`: narwhals' `tests/conftest.py` skips its
own constructor parametrization when `--use-external-constructor` is passed, and
the `narwhals_dict.testing` pytest plugin injects a plain-dict constructor
instead. Tests that are known to fail (mostly `not_implemented` functionality)
are listed in `known_failures.txt` and excluded via `--deselect`, keyed by full
`file::name` node id so that same-named tests in different files are not
conflated.

Usage, from anywhere inside the repository:

    python packages/narwhals-dict/run_tests.py             # excludes known failures
    python packages/narwhals-dict/run_tests.py --all       # includes known failures
    python packages/narwhals-dict/run_tests.py --update    # regenerate known_failures.txt
    python packages/narwhals-dict/run_tests.py -x -q ...   # extra args pass through to pytest

Fixing a `not_implemented` method? Run with `--update` afterwards to shrink the list.
`--update` prints what it added and removed, and exits non-zero on any addition,
so a new failure cannot be absorbed into the baseline unnoticed.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]
KNOWN_FAILURES = PACKAGE_ROOT / "known_failures.txt"

BASE_COMMAND = [
    "uv",
    "run",
    "pytest",
    "tests",
    "-p",
    "narwhals_dict.testing",
    "--use-external-constructor",
]

# The parametrization id is part of the node id, so one failing case does not
# deselect every other case of the same test.
FAILED_PATTERN = re.compile(
    r"^(?:FAILED|ERROR) (tests/\S+?::\w+(?:\[[^\]]*\])?)", re.MULTILINE
)


def read_known_failures(*, required: bool) -> list[str]:
    if KNOWN_FAILURES.exists():
        return KNOWN_FAILURES.read_text(encoding="utf-8").split()
    if required:
        msg = (
            f"{KNOWN_FAILURES} is missing; run with --update to generate it, "
            f"or --all to skip deselection."
        )
        raise SystemExit(msg)
    return []


def run_tests(extra_args: list[str], *, deselect_known_failures: bool) -> int:
    command = [*BASE_COMMAND, *extra_args]
    if deselect_known_failures:
        for node_id in read_known_failures(required=True):
            command.extend(["--deselect", node_id])
    return subprocess.run(command, check=False, cwd=REPO_ROOT).returncode  # noqa: S603


def update_known_failures() -> int:
    command = [*BASE_COMMAND, "-p", "no:randomly", "--tb=no", "-q"]
    result = subprocess.run(  # noqa: S603
        command, check=False, cwd=REPO_ROOT, capture_output=True, text=True
    )
    # Only rc 0 (all passed) and rc 1 (some failed) describe the suite. Anything
    # else is a crash, and rewriting the baseline from it would destroy it.
    if result.returncode not in {0, 1}:
        sys.stdout.write(result.stdout[-4000:])
        msg = f"pytest exited {result.returncode}; {KNOWN_FAILURES.name} left untouched."
        raise SystemExit(msg)

    failures = sorted(set(FAILED_PATTERN.findall(result.stdout)))
    previous = set(read_known_failures(required=False))
    added = sorted(set(failures) - previous)
    removed = sorted(previous - set(failures))
    KNOWN_FAILURES.write_text("\n".join(failures) + "\n", encoding="utf-8")

    for node_id in removed:
        sys.stdout.write(f"  fixed: {node_id}\n")
    for node_id in added:
        sys.stdout.write(f"  NEW:   {node_id}\n")
    sys.stdout.write(f"Updated {KNOWN_FAILURES.name} with {len(failures)} entries.\n")
    if added:
        sys.stdout.write(
            f"{len(added)} new failure(s) added to the baseline -- review them.\n"
        )
    return 1 if added else 0


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--update" in args:
        raise SystemExit(update_known_failures())
    include_known = "--all" in args
    args = [arg for arg in args if arg != "--all"]
    raise SystemExit(run_tests(args, deselect_known_failures=not include_known))
