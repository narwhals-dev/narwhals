from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pytest_codspeed.plugin import BenchmarkFixture


def test_execute_scripts(benchmark: BenchmarkFixture) -> None:
    root = Path(__file__).resolve().parent.parent
    # directory containing all the queries
    execute_dir = root / "execute"

    for script_path in execute_dir.glob("q[1-9]*.py"):
        print(f"executing query {script_path.stem}")  # noqa: T201
        result = benchmark(
            subprocess.run,
            [sys.executable, "-m", f"execute.{script_path.stem}"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert (
            result.returncode == 0
        ), f"Script {script_path} failed with error: {result.stderr}"
