"""Check that `output.txt` doesn't contain the string "exited with errors".

If it does, exit with status 1.

This is just used in CI.
"""

from __future__ import annotations

import sys
from pathlib import Path

if "exited with errors" in Path("output.txt").read_text("utf-8"):
    sys.exit(1)
sys.exit(0)
