from __future__ import annotations

import pytest

import narwhals.stable.v1 as nw
from tests.utils import Constructor
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data

data = {"a": [1, 2, None, 4]}
expected = {
    "cum_sum": [1, 3, None, 7],
    "reverse_cum_sum": [7, 6, None, 4],
}


@pytest.mark.parametrize("reverse", [True, False])
def test_cum_sum_expr(
    request: pytest.FixtureRequest, constructor: Constructor, *, reverse: bool
) -> None:
    if "dask" in str(constructor) and reverse:
        request.applymarker(pytest.mark.xfail)

    name = "reverse_cum_sum" if reverse else "cum_sum"
    df = nw.from_native(constructor(data))
    result = df.select(
        nw.col("a").cum_sum(reverse=reverse).alias(name),
    )

    assert_equal_data(result, {name: expected[name]})


def test_cum_sum_series(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df.select(
        cum_sum=df["a"].cum_sum(),
        reverse_cum_sum=df["a"].cum_sum(reverse=True),
    )
    assert_equal_data(result, expected)
