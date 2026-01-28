from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, get_args

from tpch.typing_ import Artifact, QueryID

if TYPE_CHECKING:
    from collections.abc import Mapping

REPO_ROOT = Path(__file__).parent.parent
TPCH_DIR = REPO_ROOT / "tpch"
DATA_DIR = TPCH_DIR / "data"
METADATA_PATH = DATA_DIR / "metadata.csv"
"""For reflection in tests.

E.g. if we *know* the query is not valid for a given `scale_factor`,
then we can determine if a failure is expected.
"""
DATABASE_TABLE_NAMES = (
    "lineitem",
    "customer",
    "nation",
    "orders",
    "part",
    "partsupp",
    "region",
    "supplier",
)
QUERY_IDS: tuple[QueryID, ...] = get_args(QueryID)
GLOBS: Mapping[Artifact, str] = {
    "database": r"*[!0-9].parquet",
    "answers": r"result_q[0-9]*.parquet",
}
LOGGER_NAME = "narwhals.tpch"
"""Per-[Logging Cookbook], pass this to `logging.getLogger(...)`.

[Logging Cookbook]: https://docs.python.org/3/howto/logging-cookbook.html#using-loggers-as-attributes-in-a-class-or-passing-them-as-parameters
"""
QUERIES_PACKAGE = "tpch.queries"
