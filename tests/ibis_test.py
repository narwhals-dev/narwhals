from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import polars as pl
import pytest

import narwhals as nw

if TYPE_CHECKING:
    import ibis

    from tests.utils import Constructor
else:
    ibis = pytest.importorskip("ibis")


@pytest.fixture
def ibis_constructor() -> Constructor:
    def func(data: dict[str, Any]) -> ibis.Table:
        df = pl.DataFrame(data)
        return ibis.memtable(df)

    return func


def test_from_native(ibis_constructor: Constructor) -> None:
    df = nw.from_native(ibis_constructor({"a": [1, 2, 3], "b": [4, 5, 6]}))
    assert df.columns == ["a", "b"]
