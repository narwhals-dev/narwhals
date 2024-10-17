from __future__ import annotations

from typing import Any

import pytest

import narwhals.stable.v1 as nw
from tests.utils import Constructor
from tests.utils import ConstructorEager
from tests.utils import compare_dicts

data = {"a": ["fdas", "edfas"]}


@pytest.mark.parametrize(
    ("offset", "length", "expected"),
    [(1, 2, {"a": ["da", "df"]}), (-2, None, {"a": ["as", "as"]})],
)
def test_str_slice(
    constructor: Constructor, offset: int, length: int | None, expected: Any
) -> None:
    df = nw.from_native(constructor(data))
    result_frame = df.select(nw.col("a").str.slice(offset, length))
    compare_dicts(result_frame, expected)


@pytest.mark.parametrize(
    ("offset", "length", "expected"),
    [(1, 2, {"a": ["da", "df"]}), (-2, None, {"a": ["as", "as"]})],
)
def test_str_slice_series(
    constructor_eager: ConstructorEager, offset: int, length: int | None, expected: Any
) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)

    result_series = df["a"].str.slice(offset, length)
    compare_dicts({"a": result_series}, expected)
