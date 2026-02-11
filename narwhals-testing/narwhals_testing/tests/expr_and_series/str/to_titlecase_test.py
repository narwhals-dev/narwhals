from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import DUCKDB_VERSION, Constructor, ConstructorEager, assert_equal_data

data = {
    "a": [
        "e.t. phone home",
        "they're bill's friends from the UK",
        "to infinity,and BEYOND!",
        "with123numbers",
        "__dunder__score_a1_.2b ?three",
    ]
}

expected = {"a": [s.title() for s in data["a"]]}


def test_str_to_titlecase_expr(
    request: pytest.FixtureRequest, constructor: Constructor
) -> None:
    if "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 2):
        reason = "version too old, duckdb 1.2 required for LambdaExpression."
        pytest.skip(reason=reason)

    if "ibis" in str(constructor):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor(data))
    result_frame = df.select(nw.col("a").str.to_titlecase())

    assert_equal_data(result_frame, expected)


def test_str_to_titlecase_series(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result_series = df["a"].str.to_titlecase()

    assert_equal_data({"a": result_series}, expected)
