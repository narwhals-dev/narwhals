from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pytest_codspeed.plugin import BenchmarkFixture

ROOT_PATH = Path(__file__).resolve().parent.parent
# Directory containing all the query scripts
QUERIES_DIR = ROOT_PATH / "queries"


def run_query(query_stem: str) -> subprocess.CompletedProcess[str]:
    """Helper function to execute the query and return the result.

    Returns:
        Subprocess result
    """
    return subprocess.run(  # noqa: S603
        [sys.executable, "-m", "execute", query_stem],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("query_path", QUERIES_DIR.glob("q[1-9]*.py"))
def test_execute_scripts(query_path: Path) -> None:
    print(f"executing query {query_path.stem}")  # noqa: T201
    result = run_query(query_stem=query_path.stem)
    assert result.returncode == 0, (
        f"Script {query_path} failed with error: {result.stderr}"
    )


@pytest.mark.parametrize("query_path", QUERIES_DIR.glob("q[1-9]*.py"))
def test_benchmark_scripts(benchmark: BenchmarkFixture, query_path: Path) -> None:
    print(f"executing query {query_path.stem}")  # noqa: T201
    result = benchmark(lambda: run_query(query_stem=query_path.stem))
    assert result.returncode == 0, (
        f"Script {query_path} failed with error: {result.stderr}"
    )
