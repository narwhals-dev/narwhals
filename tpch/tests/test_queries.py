import os
import subprocess
import sys
import unittest
from pathlib import Path


class TestQueries(unittest.TestCase):
    def test_execute_scripts(self) -> None:
        root = Path(__file__).resolve().parent.parent
        # directory containing all the queries
        execute_dir = root / "execute"

        env = os.environ.copy()
        env["PYTHONPATH"] = str(root)

        for script_path in execute_dir.glob("q[1-9]*.py"):
            result = subprocess.run(  # noqa: S603
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                env=env,
                cwd=root,
                check=False,
                shell=False,
            )
            assert (
                result.returncode == 0
            ), f"Script {script_path} failed with error: {result.stderr}"
