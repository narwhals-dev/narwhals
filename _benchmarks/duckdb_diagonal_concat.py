"""Benchmark script for DuckDB concat(how="diagonal") performance.

This script measures the performance of narwhals' nw.concat with how="diagonal"
for DuckDB relations with varying:

- Number of dataframes to concatenate
- Number of rows per dataframe
- Number of columns per dataframe
"""
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "duckdb==1.4.3",
#     "narwhals",
#     "pyarrow==21.0.0",
# ]
#
# [tool.uv.sources]
# narwhals = { path = "../" }
# ///

from __future__ import annotations

import argparse
import csv
import pathlib
import time
from dataclasses import asdict, dataclass
from itertools import product
from typing import TYPE_CHECKING, Literal

import duckdb

import narwhals as nw

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pyarrow as pa

N_ITERATIONS = 5
"""Number of iteration to run the same config, then take the average"""


@dataclass(frozen=True, slots=True, repr=True)
class BenchmarkConfig:
    """Configuration for a single benchmark run."""

    n_frames: int
    n_rows: int
    n_cols: int


@dataclass(frozen=True, slots=True, repr=True)
class BenchmarkResult:
    """Result of a single benchmark run."""

    config: BenchmarkConfig
    elapsed_time: float

    def to_dict(self) -> dict[Literal["n_frames", "n_rows", "n_cols", "time"], float]:
        return {**asdict(self.config), "time": self.elapsed_time}  # pyright: ignore[reportReturnType]


def create_relations_with_overlap(
    n_frames: int, n_rows: int, n_cols: int, overlap_fraction: float = 0.5
) -> list[duckdb.DuckDBPyRelation]:
    """Create multiple DuckDB relations with overlapping column schemas.

    This simulates a realistic diagonal concat scenario where dataframes
    share some columns but also have unique columns.

    Arguments:
        n_frames: Number of relations to create.
        n_rows: Number of rows per relation.
        n_cols: Number of columns per relation.
        overlap_fraction: Fraction of columns that overlap between consecutive relations.

    Returns:
        List of DuckDB relations with partially overlapping schemas.
    """
    relations = []
    n_overlap = max(1, int(n_cols * overlap_fraction))
    n_unique = n_cols - n_overlap

    for i in range(n_frames):
        # Common columns (shared across all dataframes)
        common_cols = ", ".join(
            f"(random() * 1000)::INTEGER AS common_{j}" for j in range(n_overlap)
        )
        # Unique columns for this dataframe
        unique_cols = ", ".join(
            f"(random() * 1000)::INTEGER AS df{i}_col_{j}" for j in range(n_unique)
        )

        all_cols = f"{common_cols}, {unique_cols}" if n_unique > 0 else common_cols
        query = f"""
            SELECT {all_cols}
            FROM generate_series(1, {n_rows})
        """  # noqa: S608
        relations.append(duckdb.sql(query))

    return relations


def _run_concat(
    relations: Sequence[duckdb.DuckDBPyRelation],
) -> tuple[nw.DataFrame[pa.Table], float]:
    """Execute the concat operation and return collected frame and elapsed time."""
    nw_frames = [nw.from_native(rel) for rel in relations]
    start = time.perf_counter()
    result = nw.concat(nw_frames, how="diagonal").collect()
    return result, time.perf_counter() - start


def run_benchmark(
    relations: Sequence[duckdb.DuckDBPyRelation], config: BenchmarkConfig
) -> BenchmarkResult:
    """Run a single benchmark iteration with optional timeout.

    Arguments:
        relations: List of DuckDB relations to concatenate.
        config: Benchmark configuration.

    Returns:
        Benchmark result with timing information.
    """
    result, elapsed = _run_concat(relations)
    assert len(result) == config.n_rows * config.n_frames  # noqa: S101
    return BenchmarkResult(config=config, elapsed_time=elapsed)


def run_benchmarks(
    n_frames_list: Sequence[int],
    n_rows_list: Sequence[int],
    n_cols_list: Sequence[int],
    output_file: str,
    overlap_fraction: float = 0.5,
) -> list[dict[str, float]]:
    """Run benchmarks across all parameter combinations.

    Arguments:
        n_frames_list: List of dataframe counts to test.
        n_rows_list: List of row counts to test.
        n_cols_list: List of column counts to test.
        output_file: Path to the output CSV file.
        overlap_fraction: Fraction of columns that overlap between dataframes.

    Returns:
        List of benchmark results.
    """
    results = []
    configs = product(n_frames_list, n_rows_list, n_cols_list)
    total_configs = len(n_frames_list) * len(n_rows_list) * len(n_cols_list)

    log = f"Running {total_configs} configurations"
    print(log)  # noqa: T201
    print("=" * len(log))  # noqa: T201

    # Write CSV header
    fieldnames = ("n_frames", "n_rows", "n_cols", "time")
    with pathlib.Path(output_file).open(encoding="utf-8", mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

    for n_frames, n_rows, n_cols in configs:
        config = BenchmarkConfig(n_frames=n_frames, n_rows=n_rows, n_cols=n_cols)

        iteration_times = []
        for _ in range(N_ITERATIONS):
            # Create fresh relations for each iteration
            relations = create_relations_with_overlap(
                n_frames, n_rows, n_cols, overlap_fraction
            )
            result = run_benchmark(relations, config)
            iteration_times.append(result.elapsed_time)

        # Report average and write to CSV
        avg_time = sum(iteration_times) / len(iteration_times)
        row = BenchmarkResult(config=config, elapsed_time=avg_time).to_dict()
        results.append(row)

        with pathlib.Path(output_file).open(
            encoding="utf-8", mode="a", newline=""
        ) as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writerow(row)

    return results


def main() -> None:
    """Run the benchmark suite."""
    parser = argparse.ArgumentParser(
        description="Benchmark DuckDB concat(how='diagonal') performance"
    )
    parser.add_argument(
        "-o",
        "--output",
        default="main.csv",
        help="Output CSV file path (default: main.csv)",
    )
    args = parser.parse_args()

    # Configuration parameters - adjust these as needed
    n_frames = (2, 5, 10, 20, 30)
    n_rows = (100, 1_000, 10_000, 100_000)
    n_cols = (6, 10, 20, 30)

    run_benchmarks(
        n_frames_list=n_frames,
        n_rows_list=n_rows,
        n_cols_list=n_cols,
        output_file=args.output,
        overlap_fraction=0.5,  # 50% column overlap between dataframes
    )


if __name__ == "__main__":
    main()
