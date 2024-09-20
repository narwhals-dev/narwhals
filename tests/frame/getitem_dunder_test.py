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
        # TODO(mikeweltevrede): Use constructor with eager
        return nw.from_native(pd.DataFrame(data), eager_only=True)

    @pytest.mark.parametrize(
        ("row_idx", "col_idx"),
        [
            ([0, 2], [0]),
            ((0, 2), [0]),
            ([0, 2], (0,)),
            ((0, 2), (0,)),
            ([0, 2], range(1)),
            (range(2), [0]),
            (range(2), range(1)),
        ],
    )
    def test_get_item_works_with_tuple_and_list_indexing(
        self,
        pd_df: nw.DataFrame[Any],
        row_idx: list[int] | tuple[int] | range,
        col_idx: list[int] | tuple[int] | range,
    ) -> None:
        pd_df = nw.from_native(pd_df, eager_only=True)
        pd_df[row_idx, col_idx]

    @pytest.mark.parametrize(
        ("row_idx", "col"),
        [
            ([0, 2], slice(1)),
            ((0, 2), slice(1)),
            (range(2), slice(1)),
        ],
    )
    def test_get_item_works_with_tuple_and_list_indexing_and_slice(
        self,
        pd_df: nw.DataFrame[Any],
        row_idx: list[int] | tuple[int] | range,
        col: slice,
    ) -> None:
        pd_df = nw.from_native(pd_df, eager_only=True)
        pd_df[row_idx, col]

    @pytest.mark.parametrize(
        ("row_idx", "col"),
        [
            ([0, 2], "x"),
            ((0, 2), "x"),
            (range(2), "x"),
        ],
    )
    def test_get_item_works_with_tuple_and_list_indexing_and_str(
        self,
        pd_df: nw.DataFrame[Any],
        row_idx: list[int] | tuple[int] | range,
        col: str,
    ) -> None:
        pd_df = nw.from_native(pd_df, eager_only=True)
        pd_df[row_idx, col]
