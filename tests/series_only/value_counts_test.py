import sys
from typing import Any

import pytest

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts

data = {"a": [4, 4, 6, 4, 1, 1]}


def test_value_counts(request: Any, constructor: Any) -> None:
    if "pandas_series_nullable" in str(constructor) or sys.version_info < (
        3,
        9,
    ):  # fails for py3.8
        request.applymarker(pytest.mark.xfail)

    series = nw.from_native(constructor(data), eager_only=True)["a"]

    sorted_result = series.value_counts(sort=True)
    expected = {"a": [4, 1, 6], "count": [3, 2, 1]}
    compare_dicts(sorted_result, expected)

    unsorted_result = series.value_counts(sort=False).sort("count", descending=True)
    compare_dicts(unsorted_result, expected)
