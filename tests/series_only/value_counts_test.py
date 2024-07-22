from __future__ import annotations

import sys
from typing import Any

import pytest

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts

data = [4, 4, 4, 1, 6, 6, 4, 4, 1, 1]


@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("name", [None, "count_name"])
def test_value_counts(
    request: Any, constructor_series: Any, normalize: Any, name: str | None
) -> None:
    if "pandas_series_nullable_constructor" in str(
        constructor_series
    ) and sys.version_info < (
        3,
        9,
    ):  # fails for py3.8
        request.applymarker(pytest.mark.xfail)

    expected_count = [5, 3, 2]
    expected_index = [4, 1, 6]

    if normalize:
        expected_count = [v / len(data) for v in expected_count]  # type: ignore[misc]

    expected_name = name or ("proportion" if normalize else "count")
    expected = {"a": expected_index, expected_name: expected_count}

    series = nw.from_native(constructor_series(data), series_only=True).alias("a")

    sorted_result = series.value_counts(sort=True, name=name, normalize=normalize)
    compare_dicts(sorted_result, expected)

    unsorted_result = series.value_counts(
        sort=False, name=name, normalize=normalize
    ).sort(expected_name, descending=True)
    compare_dicts(unsorted_result, expected)
