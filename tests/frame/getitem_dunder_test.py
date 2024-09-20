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
        return nw.from_native(pd.DataFrame(data), eager_only=True)

    def test_get_item_list_row_list_col(self, pd_df: nw.DataFrame[Any]) -> None:
        assert pd_df[[0, 2], [0]].shape == (2, 1)
