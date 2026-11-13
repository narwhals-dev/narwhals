#!/usr/bin/env python3
"""Run Narwhals' own test suite against `narwhals-dask`.

With `narwhals-dask` installed, Narwhals routes every Dask object through the plugin, so
the test suite's `dask` constructor exercises it directly. The eager constructors are
included too, for tests which go through `.lazy(backend="dask")`.

The tests run in this package's own environment (`packages/narwhals-dask/.venv`), never
in the repository's: installing `narwhals-dask` there would replace the built-in Dask
backend, which the main test suite still covers.

Usage, from anywhere inside the repository:

    python packages/narwhals-dask/run_tests.py             # whole test suite
    python packages/narwhals-dask/run_tests.py -x -q ...   # extra args pass through to pytest
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]

COMMAND = [
    "uv",
    "run",
    "--project",
    str(PACKAGE_ROOT),
    "pytest",
    "tests",
    "--constructors=dask,pandas,pyarrow,polars[eager]",
]


def main(extra_args: list[str]) -> int:
    env = os.environ.copy()
    # NOTE: Pin the environment, so that neither an active virtualenv nor a global
    # `UV_PROJECT_ENVIRONMENT` gets synced to this package's dependencies.
    env["UV_PROJECT_ENVIRONMENT"] = str(PACKAGE_ROOT / ".venv")
    env.pop("VIRTUAL_ENV", None)
    return subprocess.run(  # noqa: S603
        [*COMMAND, *extra_args], check=False, cwd=REPO_ROOT, env=env
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
