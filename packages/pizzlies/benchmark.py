#!/usr/bin/env python3
"""Benchmark `pizzlies`' import time and TPC-H queries against other backends.

Produces the numbers in the README's "Size, import time and performance" section. The
README's import times are from wheel installs (editable installs import more slowly).
Run it in this package's environment (which has pandas, Polars and PyArrow), after
generating the TPC-H data from the repository root with the root environment:

    python -m tpch.generate_data --scale-factor 0.1
    packages/pizzlies/.venv/bin/python packages/pizzlies/benchmark.py --scale-factor 0.1

Each query is timed end to end, including IO, best of `--repeat`: `[lazy]` backends run
`scan_parquet`, the query and `collect`, and `[eager]` ones `read_parquet` and the query.
Every result is checked against the reference answers.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

# Tables each query reads, in the order its `query` function takes them.
QUERY_TABLES: dict[int, tuple[str, ...]] = {
    1: ("lineitem",),
    2: ("region", "nation", "supplier", "part", "partsupp"),
    3: ("customer", "lineitem", "orders"),
    4: ("lineitem", "orders"),
    5: ("region", "nation", "customer", "lineitem", "orders", "supplier"),
    6: ("lineitem",),
    7: ("nation", "customer", "lineitem", "orders", "supplier"),
    8: ("part", "supplier", "lineitem", "orders", "customer", "nation", "region"),
    9: ("part", "partsupp", "nation", "lineitem", "orders", "supplier"),
    10: ("customer", "nation", "lineitem", "orders"),
    11: ("nation", "partsupp", "supplier"),
    12: ("lineitem", "orders"),
    13: ("customer", "orders"),
    14: ("lineitem", "part"),
    15: ("lineitem", "supplier"),
    16: ("part", "partsupp", "supplier"),
    17: ("lineitem", "part"),
    18: ("customer", "lineitem", "orders"),
    19: ("lineitem", "part"),
    20: ("part", "partsupp", "nation", "lineitem", "supplier"),
    21: ("lineitem", "nation", "orders", "supplier"),
    22: ("customer", "orders"),
}


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


def _run(query: Any, paths: list[str], backend: str) -> Any:
    import narwhals as nw

    name, _, mode = backend.partition("[")
    if mode == "lazy]":
        return query(*(nw.scan_parquet(path, backend=name) for path in paths)).collect()
    if mode == "eager]":
        return query(*(nw.read_parquet(path, backend=name) for path in paths))
    msg = f"Expected e.g. `{name}[lazy]` or `{name}[eager]`, got: {backend!r}."
    raise SystemExit(msg)


def tpch(scale_factor: str, backends: list[str], repeat: int) -> None:
    sys.path.insert(0, str(REPO_ROOT))
    import polars as pl
    from polars.testing import assert_frame_equal

    data_dir = REPO_ROOT / "tpch" / "data" / f"sf{scale_factor}"
    if not (data_dir / "lineitem.parquet").exists():
        msg = f"No TPC-H data in {data_dir}, see this script's docstring."
        raise SystemExit(msg)

    timings: dict[str, list[float]] = {backend: [] for backend in backends}
    _out()
    _out(f"TPC-H at scale factor {scale_factor}, best of {repeat} (seconds):")
    _out("       " + "".join(f"{backend:>17s}" for backend in backends))
    for query_id, tables in QUERY_TABLES.items():
        query = importlib.import_module(f"tpch.queries.q{query_id}").query
        paths = [(data_dir / f"{table}.parquet").as_posix() for table in tables]
        row = f"  q{query_id:<4}"
        for backend in backends:
            best = float("inf")
            for _ in range(repeat):
                gc.collect()
                start = time.perf_counter()
                result = _run(query, paths, backend)
                best = min(best, time.perf_counter() - start)
            timings[backend].append(best)
            row += f"{best:17.3f}"
            expected = pl.read_parquet(data_dir / f"result_q{query_id}.parquet")
            actual = pl.DataFrame(result.to_dict(as_series=False))
            assert_frame_equal(expected, actual, check_dtypes=False)
        _out(row)

    reference = backends[0]
    _out()
    _out(f"  {'':17s}{'total':>10s}{f'vs {reference}':>20s}  (geometric mean)")
    for backend, times in timings.items():
        ratio = statistics.geometric_mean(
            t / ref for t, ref in zip(times, timings[reference], strict=True)
        )
        _out(f"  {backend:17s}{sum(times):10.2f}{ratio:19.1f}x")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.partition("\n")[0])
    parser.add_argument("--scale-factor", default="0.1")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument(
        "--backends",
        default="polars[lazy],polars[eager],pizzlies[lazy],pizzlies[eager]",
        help=(
            "Comma-separated, each `<backend>[lazy]` or `<backend>[eager]`; "
            "ratios are relative to the first one."
        ),
    )
    parser.add_argument("--import-runs", type=int, default=40)
    args = parser.parse_args()
    backends = args.backends.split(",")
    import_times(
        ["pizzlies", "narwhals", "polars", "pyarrow", "pandas"], args.import_runs
    )
    tpch(args.scale_factor, backends, args.repeat)


if __name__ == "__main__":
    main()
