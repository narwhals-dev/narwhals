from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

import narwhals as nw


class TestGetItemWorksForTupleIndexing:
    """Tests added to address issue #990 - https://github.com/narwhals-dev/narwhals/issues/990"""

    @pytest.fixture
    def data(self) -> dict[str, list[int]]:
        return {"x": [1, 2, 3]}

    @pytest.fixture
    def pd_df(self, data: dict[str, list[int]]) -> nw.DataFrame[Any]:
        # TODO(mikeweltevrede): We would want to apply this on the base DF or param it over all supported DFs
        # TODO(mikeweltevrede): Use constructor with eager
        return nw.from_native(pd.DataFrame(data), eager_only=True)

    @pytest.mark.parametrize(
        ("row_idx", "col_idx"),
        [
            ([0, 2], [0]),
            ((0, 2), [0]),
            ([0, 2], (0,)),
            ((0, 2), (0,)),
        ],
    )
    def test_get_item(
        self,
        pd_df: nw.DataFrame[Any],
        row_idx: list[int] | tuple[int],
        col_idx: list[int] | tuple[int],
    ) -> None:
        pd_df = nw.from_native(pd_df, eager_only=True)
        pd_df[row_idx, col_idx]
