from __future__ import annotations

import pytest

import narwhals.stable.v1 as nw
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data

data = [1, 2, 3, 2, 1]


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        (0, None),
        (1, float("nan")),
        (2, 0.0),
        (5, 0.343622),
    ],
)
def test_skew_series(
    constructor_eager: ConstructorEager, size: int, expected: float | None
) -> None:
    result = (
        nw.from_native(constructor_eager({"a": data}), eager_only=True)
        .head(size)["a"]
        .skew()
    )
    assert_equal_data({"a": [result]}, {"a": [expected]})
