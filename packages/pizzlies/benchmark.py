#!/usr/bin/env python3
"""Benchmark `pizzlies`' import time against other backends.

Produces the import times in the README's "Size and import time" section. Run it in an
environment where `pizzlies`, Narwhals, pandas, Polars and PyArrow are installed from
wheels (editable installs import more slowly):

    python packages/pizzlies/benchmark.py
"""

from __future__ import annotations

import argparse
import statistics
import subprocess
import sys


def _out(line: str = "") -> None:
    sys.stdout.write(f"{line}\n")
    sys.stdout.flush()


IMPORT_SNIPPET = (
    "import time; t = time.perf_counter(); import {}; print(time.perf_counter() - t)"
)


def import_times(modules: list[str], runs: int) -> None:
    _out(f"Import time, median of {runs} fresh interpreters:")
    for module in modules:
        # Warm up the bytecode and OS file caches first.
        subprocess.run([sys.executable, "-c", f"import {module}"], check=True)  # noqa: S603
        times = [
            float(
                subprocess.run(  # noqa: S603
                    [sys.executable, "-c", IMPORT_SNIPPET.format(module)],
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout
            )
            for _ in range(runs)
        ]
        _out(f"  {module:10s} {statistics.median(times) * 1e3:7.1f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.partition("\n")[0])
    parser.add_argument("--import-runs", type=int, default=40)
    args = parser.parse_args()
    import_times(
        ["pizzlies", "narwhals", "polars", "pyarrow", "pandas"], args.import_runs
    )


if __name__ == "__main__":
    main()
