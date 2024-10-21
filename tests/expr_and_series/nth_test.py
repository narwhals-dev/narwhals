from __future__ import annotations

import polars as pl
import pytest

import narwhals.stable.v1 as nw
from tests.utils import Constructor
from tests.utils import assert_equal_data

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
    request: pytest.FixtureRequest,
    polars_version: tuple[int, ...],
) -> None:
    if "polars" in str(constructor) and polars_version < (1, 0, 0):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor(data))
    result = df.select(nw.nth(idx))
    assert_equal_data(result, expected)


def test_nth_not_supported(
    polars_version: tuple[int, ...],
) -> None:  # pragma: no cover
    if polars_version >= (1, 0, 0):
        pytest.skip(reason="1.0.0")
    df = nw.from_native(pl.DataFrame(data))
    with pytest.raises(
        AttributeError, match="`nth` is only supported for Polars>=1.0.0."
    ):
        df.select(nw.nth(0))
