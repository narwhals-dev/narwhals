from typing import Any

import pandas as pd
import pytest

import narwhals.stable.v1 as nw
from narwhals.utils import parse_version
from tests.utils import compare_dicts

data = {"a": [4, 4, 6, 4, 1, 1]}


def test_value_counts(request: Any, constructor: Any) -> None:
    if "pandas_nullable_constructor" in str(constructor) and parse_version(
        pd.__version__
    ) < (2, 2):
        # bug in old pandas
        request.applymarker(pytest.mark.xfail)

    series = nw.from_native(constructor(data), eager_only=True)["a"]

    sorted_result = series.value_counts(sort=True)
    expected = {"a": [4, 1, 6], "count": [3, 2, 1]}
    compare_dicts(sorted_result, expected)

    unsorted_result = series.value_counts(sort=False).sort("count", descending=True)
    compare_dicts(unsorted_result, expected)
