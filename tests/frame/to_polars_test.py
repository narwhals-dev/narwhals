from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw

if TYPE_CHECKING:
    from narwhals.testing.typing import Data
    from tests.utils import ConstructorEager

pytest.importorskip("polars")
import polars as pl


@pytest.mark.filterwarnings("ignore:.*Passing a BlockManager.*:DeprecationWarning")
def test_convert_polars(nw_eager_constructor: ConstructorEager) -> None:
    pytest.importorskip("pyarrow")
    from polars.testing import assert_frame_equal

    data: Data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.1, 8.0, 9.0]}
    df_raw = nw_eager_constructor(data)
    result = nw.from_native(df_raw).to_polars()

    expected = pl.DataFrame(data)

    assert_frame_equal(result, expected)
