from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals.stable.v1 as nw

if TYPE_CHECKING:
    from tests.utils import ConstructorEager

pytest.importorskip("polars")
import polars as pl  # noqa: E402
from polars.testing import assert_series_equal  # noqa: E402

data = [1, 3, 2]


def test_series_to_polars(constructor_eager: ConstructorEager) -> None:
    result = (
        nw.from_native(constructor_eager({"a": data}), eager_only=True)["a"]
        .alias("a")
        .to_polars()
    )

    expected = pl.Series("a", data)
    assert_series_equal(result, expected)
