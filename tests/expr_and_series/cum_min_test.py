from __future__ import annotations

import pytest

import narwhals.stable.v1 as nw
from tests.utils import Constructor
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data

data = {"a": [3, 1, None, 2]}

expected = {
    "cum_min": [3, 1, None, 1],
    "reverse_cum_min": [1, 1, None, 2],
}


def test_cum_min_expr(request: pytest.FixtureRequest, constructor: Constructor) -> None:
    if "dask" in str(constructor):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor(data))
    result = df.select(
        cum_min=nw.col("a").cum_min(),
        reverse_cum_min=nw.col("a").cum_min(reverse=True),
    )

    assert_equal_data(result, expected)


def test_cum_min_series(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df.select(
        cum_min=df["a"].cum_min(),
        reverse_cum_min=df["a"].cum_min(reverse=True),
    )
    assert_equal_data(result, expected)
