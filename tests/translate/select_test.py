from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest

import narwhals.stable.v1 as nw
from narwhals.dependencies import get_polars

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
    pl = get_polars()
    pl_version = tuple(int(i) for i in pl.build_info().get("version").split("."))
    if pl_version < (0, 20, 26):
        with pytest.raises(
            AttributeError, match="`nth` is only supported for Polars>=0.20.26."
        ):
            result = func(df)
    else:
        result = func(df)
        pd.testing.assert_frame_equal(result, pd.DataFrame(expected))
