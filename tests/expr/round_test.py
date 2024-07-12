from __future__ import annotations

from typing import Any

import pytest

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts


@pytest.mark.parametrize("decimals", [0, 1, 2])
def test_round(constructor: Any, decimals: int) -> None:
    data = {"a": [1.12345, 2.56789, 3.901234]}
    df_raw = constructor(data)
    df = nw.from_native(df_raw, eager_only=True)

    expected_data = {k: [round(e, decimals) for e in v] for k, v in data.items()}
    result_frame = df.select(nw.col("a").round(decimals))
    compare_dicts(result_frame, expected_data)

    result_series = df["a"].round(decimals)

    assert result_series.to_numpy().tolist() == expected_data["a"]
