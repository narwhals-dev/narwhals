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
# ruff: noqa: S608,T201
from __future__ import annotations

import argparse
import csv
import pathlib
import time
from dataclasses import asdict, dataclass
from itertools import product
from typing import TYPE_CHECKING, ClassVar, Literal

import duckdb

import narwhals as nw

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

    import pyarrow as pa

N_ITERATIONS = 5
"""Number of iteration to run the same config, then take the average"""


@dataclass(frozen=True, slots=True, repr=True)
class BenchmarkConfig:
    """Configuration for a single benchmark run."""

    n_frames: int
    n_rows: int
    n_cols: int
    overlap_fraction: ClassVar[float] = 0.5
    """50% column overlap between dataframes"""

    @property
    def n_overlap(self) -> int:
        return max(1, int(self.n_cols * self.overlap_fraction))

    @property
    def n_unique(self) -> int:
        return self.n_cols - self.n_overlap

    def _generate_relations(self) -> Iterator[duckdb.DuckDBPyRelation]:
        """Create multiple DuckDB relations with overlapping column schemas.

        This simulates a realistic diagonal concat scenario where dataframes
        share some columns but also have unique columns.
        """
        n_overlap = self.n_overlap
        n_unique = self.n_unique

        for i in range(self.n_frames):
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
                FROM generate_series(1, {self.n_rows})
            """
            yield duckdb.sql(query)

    def _run_benchmark(self) -> float:
        result, elapsed = _run_concat(self._generate_relations())
        expected_length = self.n_rows * self.n_frames
        if len(result) != expected_length:
            msg = f"Expected result length {expected_length!r}, got: {len(result)!r}"
            raise ValueError(msg)
        return elapsed

    def run_benchmark(self, n_iterations: int = 5) -> BenchmarkResult:
        if n_iterations > 1:
            iteration_times = [self._run_benchmark() for _ in range(n_iterations)]
            avg_time = sum(iteration_times) / len(iteration_times)
            return self.to_result(avg_time)
        return self.to_result(self._run_benchmark())

    def to_result(self, elapsed_time: float) -> BenchmarkResult:
        return BenchmarkResult(self, elapsed_time)


def iter_configs(
    n_frames_list: Sequence[int], n_rows_list: Sequence[int], n_cols_list: Sequence[int]
) -> Iterator[BenchmarkConfig]:
    for n_frames, n_rows, n_cols in product(n_frames_list, n_rows_list, n_cols_list):
        yield BenchmarkConfig(n_frames, n_rows, n_cols)


@dataclass(frozen=True, slots=True, repr=True)
class BenchmarkResult:
    """Result of a single benchmark run."""

    config: BenchmarkConfig
    elapsed_time: float

    def to_dict(self) -> dict[Literal["n_frames", "n_rows", "n_cols", "time"], float]:
        return {**asdict(self.config), "time": self.elapsed_time}  # pyright: ignore[reportReturnType]


def _run_concat(
    relations: Iterable[duckdb.DuckDBPyRelation],
) -> tuple[nw.DataFrame[pa.Table], float]:
    """Execute the concat operation and return collected frame and elapsed time."""
    nw_frames = [nw.from_native(rel) for rel in relations]
    start = time.perf_counter()
    result = nw.concat(nw_frames, how="diagonal").collect()
    return result, time.perf_counter() - start


def run_benchmarks(
    n_frames_list: Sequence[int],
    n_rows_list: Sequence[int],
    n_cols_list: Sequence[int],
    output: pathlib.Path,
) -> None:
    """Run benchmarks across all parameter combinations.

    Arguments:
        n_frames_list: List of dataframe counts to test.
        n_rows_list: List of row counts to test.
        n_cols_list: List of column counts to test.
        output: Path to the output CSV file.
    """
    total_configs = len(n_frames_list) * len(n_rows_list) * len(n_cols_list)
    log = f"Running {total_configs} configurations"
    underline = "=" * len(log)
    print(f"{log}\n{underline}")

    # Write CSV header
    fieldnames = ("n_frames", "n_rows", "n_cols", "time")
    with output.open(encoding="utf-8", mode="w", newline="") as file:
        csv.DictWriter(file, fieldnames).writeheader()

    # Report average and write to CSV
    rows = (
        config.run_benchmark().to_dict()
        for config in iter_configs(n_frames_list, n_rows_list, n_cols_list)
    )
    with output.open(encoding="utf-8", mode="a", newline="") as file:
        csv.DictWriter(file, fieldnames).writerows(rows)


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

    run_benchmarks(n_frames, n_rows, n_cols, pathlib.Path(args.output))


if __name__ == "__main__":
    main()
