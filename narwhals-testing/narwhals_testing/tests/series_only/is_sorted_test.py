from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import ConstructorEager, assert_equal_data

data = [1, 3, 2]
data_dups = [4, 4, 6]
data_sorted = [7.0, 8.0, 9.0]


@pytest.mark.parametrize(
    ("input_data", "descending", "expected"),
    [(data, False, False), (data_sorted, False, True), (data_sorted, True, False)],
)
def test_is_sorted(
    constructor_eager: ConstructorEager,
    input_data: str,
    descending: bool,  # noqa: FBT001
    expected: bool,  # noqa: FBT001
) -> None:
    series = nw.from_native(constructor_eager({"a": input_data}), eager_only=True)["a"]
    result = series.is_sorted(descending=descending)
    assert_equal_data({"a": [result]}, {"a": [expected]})


def test_is_sorted_invalid(constructor_eager: ConstructorEager) -> None:
    series = nw.from_native(constructor_eager({"a": data_sorted}), eager_only=True)["a"]

    with pytest.raises(TypeError):
        series.is_sorted(descending="invalid_type")  # type: ignore[arg-type]
