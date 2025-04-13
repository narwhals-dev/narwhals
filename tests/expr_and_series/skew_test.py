from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data

data = [1, 2, 3, 2, 1]


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ([], None),
        ([1], None),
        ([1, 2], 0.0),
        ([0.0, 0.0, 0.0], None),
        ([1, 2, 3, 2, 1], 0.343622),
    ],
)
def test_skew_series(
    constructor_eager: ConstructorEager, data: list[float], expected: float | None
) -> None:
    result = nw.from_native(constructor_eager({"a": data}), eager_only=True)["a"].skew()
    assert_equal_data({"a": [result]}, {"a": [expected]})
