from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT_PATH = Path(__file__).resolve().parent.parent
# Directory containing all the query scripts
QUERIES_DIR = ROOT_PATH / "queries"


@pytest.mark.parametrize("query_path", QUERIES_DIR.glob("q[1-9]*.py"))
def test_execute_scripts(query_path: Path) -> None:
    print(f"executing query {query_path.stem}")  # noqa: T201
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "execute", str(query_path.stem)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"Script {query_path} failed with error: {result.stderr}"
    )
