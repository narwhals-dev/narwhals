from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
TPCH_DIR = REPO_ROOT / "tpch"
DATA_DIR = TPCH_DIR / "data"
METADATA_PATH = DATA_DIR / "metadata.csv"
"""For reflection in tests.

E.g. if we *know* the query is not valid for a given `scale_factor`,
then we can determine if a failure is expected.
"""
