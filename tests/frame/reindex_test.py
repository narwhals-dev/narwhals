from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts

data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}


@pytest.mark.parametrize("df_raw", [pd.DataFrame(data)])
def test_reindex(df_raw: Any) -> None:
    df = nw.from_native(df_raw, eager_only=True)
    result = df.select("b", df["a"].sort(descending=True))
    expected = {"b": [4, 4, 6], "a": [3, 2, 1]}
    compare_dicts(result, expected)
    result = df.select("b", nw.col("a").sort(descending=True))
    compare_dicts(result, expected)

    s = df["a"]
    result_s = s > s.sort()
    assert not result_s[0]
    assert result_s[1]
    assert not result_s[2]
    result = df.with_columns(s.sort())
    expected = {"a": [1, 2, 3], "b": [4, 4, 6], "z": [7.0, 8.0, 9.0]}  # type: ignore[list-item]
    compare_dicts(result, expected)
    with pytest.raises(ValueError, match="Multi-output expressions are not supported"):
        nw.to_native(df.with_columns(nw.all() + nw.all()))
