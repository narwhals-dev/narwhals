from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw

if TYPE_CHECKING:
    from tests.utils import ConstructorEager

pytest.importorskip("polars")
import polars as pl

data = [1, 3, 2]


def test_series_to_polars(nw_eager_constructor: ConstructorEager) -> None:
    pytest.importorskip("pyarrow")
    from polars.testing import assert_series_equal

    result = (
        nw.from_native(nw_eager_constructor({"a": data}), eager_only=True)["a"]
        .alias("a")
        .to_polars()
    )

    expected = pl.Series("a", data)
    assert_series_equal(result, expected)
