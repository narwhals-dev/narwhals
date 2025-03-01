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
def test_cum_sum_expr(constructor_eager: ConstructorEager, *, reverse: bool) -> None:
    name = "reverse_cum_sum" if reverse else "cum_sum"
    df = nw.from_native(constructor_eager(data))
    result = df.select(
        nw.col("a").cum_sum(reverse=reverse).alias(name),
    )

    assert_equal_data(result, {name: expected[name]})


def test_lazy_cum_sum(constructor: Constructor, request: pytest.FixtureRequest) -> None:
    if "duckdb" in str(constructor):
        # no window function support yet in duckdb
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(
        constructor({"a": [1, 2, 3], "b": [1, 0, 2], "i": [0, 1, 2], "g": [1, 1, 1]})
    )
    result = df.with_columns(nw.col("a").cum_sum().over("g", _order_by="b")).sort("i")
    expected = {"a": [3, 2, 6], "b": [1, 0, 2], "i": [0, 1, 2]}
    assert_equal_data(result, expected)


def test_cum_sum_series(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df.select(
        cum_sum=df["a"].cum_sum(),
        reverse_cum_sum=df["a"].cum_sum(reverse=True),
    )
    assert_equal_data(result, expected)
