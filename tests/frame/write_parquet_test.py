from __future__ import annotations

import os
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
    path = str(tmpdir / "foo.parquet")  # type: ignore[operator]
    nw.from_native(constructor(data), eager_only=True).write_parquet(path)
    assert os.path.exists(path)
