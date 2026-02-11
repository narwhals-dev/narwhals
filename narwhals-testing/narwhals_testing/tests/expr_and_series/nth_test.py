from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw
from tests.utils import POLARS_VERSION, Constructor, assert_equal_data

if TYPE_CHECKING:
    from collections.abc import Mapping

data: Mapping[str, Any] = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.1, 8.0, 9.0]}


@pytest.mark.parametrize(
    ("idx", "expected"),
    [
        (0, {"a": [1, 3, 2]}),
        ([0, 1], {"a": [1, 3, 2], "b": [4, 4, 6]}),
        ([0, 2], {"a": [1, 3, 2], "z": [7.1, 8.0, 9.0]}),
    ],
)
def test_nth(
    constructor: Constructor, idx: int | list[int], expected: dict[str, list[int]]
) -> None:
    if "polars" in str(constructor) and POLARS_VERSION < (1, 0, 0):
        pytest.skip()
    df = nw.from_native(constructor(data))
    result = df.select(nw.nth(idx))
    assert_equal_data(result, expected)


@pytest.mark.skipif(POLARS_VERSION >= (1, 0, 0), reason="1.0.0")
def test_nth_not_supported() -> None:  # pragma: no cover
    pytest.importorskip("polars")
    import polars as pl

    df = nw.from_native(pl.DataFrame(data))
    pattern = re.compile(r"`nth` .+ \'polars>=1.0.0\'.+`col`.+instead", re.DOTALL)
    with pytest.raises(NotImplementedError, match=pattern):
        df.select(nw.nth(0))
