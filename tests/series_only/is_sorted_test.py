from __future__ import annotations

from typing import Any

import pytest

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts

data = [1, 3, 2]
data_dups = [4, 4, 6]
data_sorted = [7.0, 8, 9]


@pytest.mark.parametrize(
    ("input_data", "descending", "expected"),
    [(data, False, False), (data_sorted, False, True), (data_sorted, True, False)],
)
def test_is_sorted(
    constructor_series: Any,
    input_data: str,
    descending: bool,  # noqa: FBT001
    expected: bool,  # noqa: FBT001
) -> None:
    series = nw.from_native(constructor_series(input_data), series_only=True)
    result = series.is_sorted(descending=descending)
    compare_dicts({"a": [result]}, {"a": [expected]})


def test_is_sorted_invalid(constructor_series: Any) -> None:
    series = nw.from_native(constructor_series(data_sorted), series_only=True)

    with pytest.raises(TypeError):
        series.is_sorted(descending="invalid_type")  # type: ignore[arg-type]
