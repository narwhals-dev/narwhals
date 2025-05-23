from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import Constructor, ConstructorEager, assert_equal_data


@pytest.mark.parametrize("decimals", [0, 1, 2])
def test_round(constructor: Constructor, decimals: int) -> None:
    data = {"a": [2.12345, 2.56789, 3.901234]}
    df_raw = constructor(data)
    df = nw.from_native(df_raw)

    expected_data = {k: [round(e, decimals) for e in v] for k, v in data.items()}
    result_frame = df.select(nw.col("a").round(decimals))
    assert_equal_data(result_frame, expected_data)


@pytest.mark.parametrize("decimals", [0, 1, 2])
def test_round_series(constructor_eager: ConstructorEager, decimals: int) -> None:
    data = {"a": [1.12345, 2.56789, 3.901234]}
    df_raw = constructor_eager(data)
    df = nw.from_native(df_raw, eager_only=True)

    expected_data = {k: [round(e, decimals) for e in v] for k, v in data.items()}
    result_series = df["a"].round(decimals)

    assert_equal_data({"a": result_series}, expected_data)
