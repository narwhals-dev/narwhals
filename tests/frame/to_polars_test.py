from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Mapping

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import narwhals.stable.v1 as nw

if TYPE_CHECKING:
    from tests.utils import ConstructorEager


@pytest.mark.filterwarnings("ignore:.*Passing a BlockManager.*:DeprecationWarning")
def test_convert_polars(constructor_eager: ConstructorEager) -> None:
    data: Mapping[str, Any] = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.1, 8.0, 9.0]}
    df_raw = constructor_eager(data)
    result = nw.from_native(df_raw).to_polars()  # type: ignore[union-attr]

    expected = pl.DataFrame(data)

    assert_frame_equal(result, expected)
