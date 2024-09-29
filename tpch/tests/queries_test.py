import subprocess
import sys
from pathlib import Path


def test_execute_scripts() -> None:
    root = Path(__file__).resolve().parent.parent
    # directory containing all the queries
    execute_dir = root / "execute"

    for script_path in execute_dir.glob("q[1-9]*.py"):
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
