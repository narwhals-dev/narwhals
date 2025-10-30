from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw
from tests.utils import assert_equal_data

if TYPE_CHECKING:
    import ibis
    import polars as pl

    from tests.utils import Constructor
else:
    ibis = pytest.importorskip("ibis")
    pl = pytest.importorskip("polars")


@pytest.fixture
def ibis_constructor() -> Constructor:
    def func(data: dict[str, Any]) -> ibis.Table:
        df = pl.DataFrame(data)
        return ibis.memtable(df)

    return func


def test_from_native(ibis_constructor: Constructor) -> None:
    df = nw.from_native(ibis_constructor({"a": [1, 2, 3], "b": [4, 5, 6]}))
    assert df.columns == ["a", "b"]


def test_is_finite_integer_column(ibis_constructor: Constructor) -> None:
    # Test for https://github.com/narwhals-dev/narwhals/issues/3255
    df = nw.from_native(ibis_constructor({"a": [1, 2, 3]}))
    result = df.select(nw.col("a").is_finite())  # should not error
    assert_equal_data(result, {"a": [True, True, True]})
