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

expected_non_alphabetic = {
    "a": [
        "E.T. Phone Home",
        "They'Re Bill'S Friends From The Uk",
        "To Infinity,And Beyond!",
        "With123Numbers",
        "__Dunder__Score_A1_.2B ?Three",
    ]
}
expected_non_alphanumeric = {
    "a": [
        "E.T. Phone Home",
        "They'Re Bill'S Friends From The Uk",
        "To Infinity,And Beyond!",
        "With123numbers",
        "__Dunder__Score_A1_.2b ?Three",
    ]
}

NON_ALPHANUMERIC_BACKENDS = ("duckdb", "polars", "pyspark")


def test_str_to_titlecase_expr(
    request: pytest.FixtureRequest, constructor: Constructor
) -> None:
    if "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 2):
        reason = "version too old, duckdb 1.2 required for LambdaExpression."
        pytest.skip(reason=reason)

    if "ibis" in str(constructor):
        request.applymarker(pytest.mark.xfail)

    expected = (
        expected_non_alphanumeric
        if any(x in str(constructor) for x in NON_ALPHANUMERIC_BACKENDS)
        else expected_non_alphabetic
    )

    df = nw.from_native(constructor(data))
    result_frame = df.select(nw.col("a").str.to_titlecase())

    assert_equal_data(result_frame, expected)


def test_str_to_titlecase_series(constructor_eager: ConstructorEager) -> None:
    expected = (
        expected_non_alphanumeric
        if any(x in str(constructor_eager) for x in NON_ALPHANUMERIC_BACKENDS)
        else expected_non_alphabetic
    )

    df = nw.from_native(constructor_eager(data), eager_only=True)
    result_series = df["a"].str.to_titlecase()

    assert_equal_data({"a": result_series}, expected)
