from __future__ import annotations

from typing import Any

import polars as pl
import pytest

import narwhals.stable.v1 as nw
from narwhals.utils import parse_version
from tests.utils import Constructor
from tests.utils import compare_dicts

data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.1, 8, 9]}


@pytest.mark.parametrize(
    ("idx", "expected"),
    [
        (0, {"a": [1, 3, 2]}),
        ([0, 1], {"a": [1, 3, 2], "b": [4, 4, 6]}),
        ([0, 2], {"a": [1, 3, 2], "z": [7.1, 8, 9]}),
    ],
)
def test_nth(
    constructor: Constructor,
    idx: int | list[int],
    expected: dict[str, list[int]],
    request: Any,
) -> None:
    if "polars" in str(constructor) and parse_version(pl.__version__) < parse_version(
        "1.0.0"
    ):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor(data))
    result = df.select(nw.nth(idx))
    compare_dicts(result, expected)


@pytest.mark.skipif(
    parse_version(pl.__version__) >= parse_version("1.0.0"),
    reason="1.0.0",
)
def test_nth_not_supported() -> None:  # pragma: no cover
    df = nw.from_native(pl.DataFrame(data))
    with pytest.raises(
        AttributeError, match="`nth` is only supported for Polars>=1.0.0."
    ):
        df.select(nw.nth(0))
