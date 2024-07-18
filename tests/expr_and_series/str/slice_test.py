from __future__ import annotations

from typing import Any

import pytest

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts

data = {"a": ["fdas", "edfas"]}


@pytest.mark.parametrize(
    ("offset", "length", "expected"),
    [(1, 2, {"a": ["da", "df"]}), (-2, None, {"a": ["as", "as"]})],
)
def test_str_slice(
    constructor: Any, offset: int, length: int | None, expected: Any
) -> None:
    df = nw.from_native(constructor(data), eager_only=True)
    result_frame = df.select(nw.col("a").str.slice(offset, length))
    compare_dicts(result_frame, expected)

    result_series = df["a"].str.slice(offset, length)
    assert result_series.to_numpy().tolist() == expected["a"]
