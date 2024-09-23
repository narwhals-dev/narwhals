from __future__ import annotations

from typing import Any

import pytest

import narwhals.stable.v1 as nw
from tests.utils import Constructor
from tests.utils import compare_dicts

data = {"a": [-1.0, 0, 1, 10, 100]}


def test_log_expr(constructor: Constructor, request: pytest.FixtureRequest) -> None:
    if "dask" in str(constructor):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor(data))
    result = df.select(nw.col("a").log10())
    expected = {"a": [float("nan"), float("-inf"), 0, 1, 2]}
    compare_dicts(result, expected)


def test_log_series(constructor_eager: Any) -> None:
    series = nw.from_native(constructor_eager(data), eager_only=True)["a"]
    result = series.log10()
    expected = {"a": [float("nan"), float("-inf"), 0, 1, 2]}
    compare_dicts({"a": result}, expected)
