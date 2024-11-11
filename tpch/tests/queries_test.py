from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT_PATH = Path(__file__).resolve().parent.parent
# directory containing all the script to execute queries
EXECUTE_DIR = ROOT_PATH / "execute"


@pytest.mark.parametrize("script_path", EXECUTE_DIR.glob("q[1-9]*.py"))
def test_execute_scripts(script_path: Path) -> None:
    print(f"executing query {script_path.stem}")  # noqa: T201
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-m", f"execute.{script_path.stem}"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert (
        result.returncode == 0
    ), f"Script {script_path} failed with error: {result.stderr}"
