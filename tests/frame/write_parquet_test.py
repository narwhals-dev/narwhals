from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import narwhals.stable.v1 as nw
from narwhals.utils import parse_version

data = {"a": [1, 2, 3]}


@pytest.mark.skipif(
    parse_version(pd.__version__) < parse_version("2.0.0"), reason="too old for pyarrow"
)
def test_write_parquet(constructor: Any, tmpdir: pytest.TempdirFactory) -> None:
    path = Path(str(tmpdir)) / "foo.parquet"
    nw.from_native(constructor(data), eager_only=True).write_parquet(path)
    assert path.exists()
