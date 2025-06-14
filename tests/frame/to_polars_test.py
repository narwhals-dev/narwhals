from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tests.utils import ConstructorEager

pytest.importorskip("polars")
import polars as pl


@pytest.mark.filterwarnings("ignore:.*Passing a BlockManager.*:DeprecationWarning")
def test_convert_polars(constructor_eager: ConstructorEager) -> None:
    from polars.testing import assert_frame_equal

    data: Mapping[str, Any] = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.1, 8.0, 9.0]}
    df_raw = constructor_eager(data)
    result = nw.from_native(df_raw).to_polars()

    expected = pl.DataFrame(data)

    assert_frame_equal(result, expected)
