"""Read the files of https://github.com/apache/parquet-testing, comparing with PyArrow.

Set `PARQUET_TESTING_DIR` to a checkout of that repository to run these tests.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
from pizzlies._plugin.parquet import read_parquet

pq = pytest.importorskip("pyarrow.parquet")

ROOT = Path(env) if (env := os.environ.get("PARQUET_TESTING_DIR")) else None
FILES = (
    sorted(
        path.relative_to(ROOT).as_posix()
        for directory in ("data", "bad_data")
        for path in (ROOT / directory).glob("*.parquet")
    )
    if ROOT is not None
    else []
)

# Files where PyArrow is not the right reference.
EXPECTED: dict[str, Any] = {
    # PyArrow 26 reads no rows, but the README says that readers "must skip these pages
    # and read three rows" (as Polars does).
    "data/empty_data_pages_v1.parquet": {"int64": [1, 2, 3]},
    "data/empty_data_pages_v2.parquet": {"int64": [1, 2, 3]},
    # PyArrow and Polars reject the optional map key, we read it anyway.
    "data/incorrect_map_schema.parquet": {
        "my_map": [
            [{"key": "parent", "value": "another"}, {"key": "name", "value": "report"}]
        ]
    },
}
RAISES: dict[str, str] = {
    # Spark-written INT96 timestamps beyond the year 9999, which PyArrow wraps around.
    "data/int96_from_spark.parquet": "can't represent",
    # The page header reports the wrong size. PyArrow reads it anyway, Polars raises.
    "data/nation.dict-malformed.parquet": "may be corrupt",
    # Deliberately corrupted, which PyArrow doesn't check by default.
    "data/datapage_v1-corrupt-checksum.parquet": "checksum mismatch",
    "data/rle-dict-uncompressed-corrupt-checksum.parquet": "checksum mismatch",
}


def _normalise(value: Any) -> Any:
    if hasattr(value, "to_pydatetime"):  # nanosecond `pd.Timestamp`s, truncated as we do
        return value.to_pydatetime(warn=False)
    if isinstance(value, list):
        return [_normalise(v) for v in value]
    if isinstance(value, tuple):  # PyArrow's map entries
        return {"key": _normalise(value[0]), "value": _normalise(value[1])}
    if isinstance(value, dict):
        return {k: _normalise(v) for k, v in value.items()}
    if value != value:  # noqa: PLR0124
        return "NaN"
    return value


@pytest.mark.skipif(ROOT is None, reason="PARQUET_TESTING_DIR is not set")
@pytest.mark.parametrize("name", FILES or ["unused"])
def test_parquet_testing(name: str) -> None:
    assert ROOT is not None
    path = str(ROOT / name)
    if name in RAISES:
        with pytest.raises(ValueError, match=RAISES[name]):
            read_parquet(path)
        return
    try:
        expected = EXPECTED.get(name) or pq.read_table(path).to_pydict()
    except Exception:  # noqa: BLE001
        # A file PyArrow rejects must be rejected as corrupt or unsupported, not crash.
        with pytest.raises((ValueError, NotImplementedError)):
            read_parquet(path)
        return
    try:
        result = read_parquet(path)
    except NotImplementedError:
        pytest.skip("uses a Parquet feature pizzlies doesn't support")
    assert {k: list(map(_normalise, v)) for k, v in result.items()} == {
        k: list(map(_normalise, v)) for k, v in expected.items()
    }
