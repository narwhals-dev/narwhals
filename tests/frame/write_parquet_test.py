from __future__ import annotations

import os
from typing import TYPE_CHECKING
from typing import Any

import narwhals as nw

data = {"a": [1, 2, 3]}

if TYPE_CHECKING:
    import pytest


def test_write_parquet(constructor: Any, tmpdir: pytest.TempdirFactory) -> None:
    path = str(tmpdir / "foo.parquet")  # type: ignore[operator]
    nw.from_native(constructor(data), eager_only=True).write_parquet(path)
    assert os.path.exists(path)
