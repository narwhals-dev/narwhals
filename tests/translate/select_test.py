from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import polars as pl
import pytest

import narwhals.stable.v1 as nw
from narwhals.utils import parse_version

if TYPE_CHECKING:
    from narwhals.typing import IntoDataFrameT

data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.1, 8, 9]}


@pytest.mark.parametrize(
    ("idx", "expected"),
    [
        (0, {"a": [1, 3, 2]}),
        ([0, 1], {"a": [1, 3, 2], "b": [4, 4, 6]}),
        ([0, 2], {"a": [1, 3, 2], "z": [7.1, 8, 9]}),
    ],
)
def test_nth(idx: int | list[int], expected: dict[str, list[int]]) -> None:
    @nw.narwhalify
    def func(df: nw.DataFrame[IntoDataFrameT]) -> nw.DataFrame[IntoDataFrameT]:
        return df.select(nw.nth(idx))

    df = pd.DataFrame(data)
    result = func(df)
    pd.testing.assert_frame_equal(result, pd.DataFrame(expected))


@pytest.mark.skipif(
    parse_version(pl.__version__) >= parse_version("0.20.26"),
    reason="only riase error for Polars versions < 0.20.26",
)
def test_nth_old_polars() -> None:  # pragma: no cover
    @nw.narwhalify
    def func(df: nw.DataFrame[IntoDataFrameT]) -> nw.DataFrame[IntoDataFrameT]:
        return df.select(nw.nth(0))

    df = pl.DataFrame(data)
    with pytest.raises(
        AttributeError, match="`nth` is only supported for Polars>=0.20.26."
    ):
        func(df)
