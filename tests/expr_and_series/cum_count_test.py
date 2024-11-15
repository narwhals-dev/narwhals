from __future__ import annotations

import pytest

import narwhals.stable.v1 as nw
from tests.utils import Constructor
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data

data = {
    "a": ["x", "y", None, "z"],
}


def test_cum_count_expr(request: pytest.FixtureRequest, constructor: Constructor) -> None:
    if "dask" in str(constructor):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor(data))
    result = df.select(
        cum_count=nw.col("a").cum_count(),
        reverse_cum_count=nw.col("a").cum_count(reverse=True),
    )
    expected = {
        "cum_count": [1, 2, 2, 3],
        "reverse_cum_count": [3, 2, 1, 1],
    }
    assert_equal_data(result, expected)


def test_cum_count_series(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df.select(
        cum_count=df["a"].cum_count(),
        reverse_cum_count=df["a"].cum_count(reverse=True),
    )
    expected = {
        "cum_count": [1, 2, 2, 3],
        "reverse_cum_count": [3, 2, 1, 1],
    }
    assert_equal_data(result, expected)
