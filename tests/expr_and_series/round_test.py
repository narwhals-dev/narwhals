from __future__ import annotations

import math
from typing import Any

import pytest

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts


def round_half_towards_infinity(n: float, decimals: int) -> float:
    rounded = round(n, decimals)
    frac, whole = math.modf(n)
    if frac == 0.5 and rounded == whole:
        return whole + 1
    return rounded


def round_per_backend(backend: Any, n: float, decimals: int) -> float:
    if backend.__name__.startswith(("polars", "pyarrow")):
        return round_half_towards_infinity(n, decimals)
    else:
        return round(n, decimals)


@pytest.mark.parametrize("decimals", [0, 1, 2])
def test_round(constructor: Any, decimals: int) -> None:
    data = {"a": [1.12345, 2.56789, 3.901234, 2.5]}
    df_raw = constructor(data)
    df = nw.from_native(df_raw)

    expected_data = {
        k: [round_per_backend(constructor, e, decimals) for e in v]
        for k, v in data.items()
    }
    result_frame = df.select(nw.col("a").round(decimals))
    compare_dicts(result_frame, expected_data)


@pytest.mark.parametrize("decimals", [0, 1, 2])
def test_round_series(constructor_eager: Any, decimals: int) -> None:
    data = {"a": [1.12345, 2.56789, 3.901234, 2.5]}
    df_raw = constructor_eager(data)
    df = nw.from_native(df_raw, eager_only=True)

    expected_data = {
        k: [round_per_backend(constructor_eager, e, decimals) for e in v]
        for k, v in data.items()
    }
    result_series = df["a"].round(decimals)

    assert result_series.to_list() == expected_data["a"]
