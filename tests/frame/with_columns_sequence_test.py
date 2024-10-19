from __future__ import annotations

import numpy as np
import pytest

import narwhals.stable.v1 as nw
from tests.utils import Constructor
from tests.utils import compare_dicts

data = {
    "a": ["foo", "bars"],
    "ab": ["foo", "bars"],
}


def test_with_columns(constructor: Constructor, request: pytest.FixtureRequest) -> None:
    if "dask" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    result = (
        nw.from_native(constructor(data))
        .with_columns(d=np.array([4, 5]))
        .with_columns(e=nw.col("d") + 1)
        .select("d", "e")
    )
    expected = {"d": [4, 5], "e": [5, 6]}
    compare_dicts(result, expected)
