from __future__ import annotations

import pytest

import pandas as pd

import narwhals.stable.v1 as nw
from tests.utils import Constructor
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data

data = {"a": [[1, 2], [3, 4, None], None, [], [None]]}
expected = {"a": [2, 3, None, 0, 1]}

def test_len_expr(
    request: pytest.FixtureRequest,
    constructor: Constructor,
) -> None:
    if "dask" in str(constructor):
        request.applymarker(pytest.mark.xfail)


    result = nw.from_native(constructor(data)).select(nw.col("a").list.len())

    assert_equal_data(result, expected)


def test_len_series(
    request: pytest.FixtureRequest,
    constructor_eager: ConstructorEager,
) -> None:
    if "dask" in str(constructor_eager):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor_eager(data), eager_only=True)

    result = df["a"].list.len()
    assert_equal_data({"a": result}, expected)
