from __future__ import annotations

from copy import deepcopy

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
expected = {
    "a": [
        "E.T. Phone Home",
        "They'Re Bill'S Friends From The Uk",
        "To Infinity,And Beyond!",
        "With123Numbers",
        "__Dunder__Score_A1_.2B ?Three",
    ]
}
REPLACEMENTS = {
    "With123Numbers": "With123numbers",
    "__Dunder__Score_A1_.2B ?Three": "__Dunder__Score_A1_.2b ?Three",
}


def test_str_to_titlecase_expr(
    request: pytest.FixtureRequest, constructor: Constructor
) -> None:
    if any(x in str(constructor) for x in ("ibis", "pyspark", "sqlframe")):
        request.applymarker(pytest.mark.xfail)

    if "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 3):
        reason = "version too old, duckdb 1.3 required for SQLExpression."
        request.applymarker(pytest.mark.xfail(reason=reason))

    expected_ = deepcopy(expected)
    if "polars" in str(constructor) or "duckdb" in str(constructor):
        expected_ = {"a": [REPLACEMENTS.get(el, el) for el in expected_["a"]]}

    df = nw.from_native(constructor(data))
    result_frame = df.select(nw.col("a").str.to_titlecase())

    assert_equal_data(result_frame, expected_)


def test_str_to_titlecase_series(constructor_eager: ConstructorEager) -> None:
    expected_ = deepcopy(expected)
    if "polars" in str(constructor_eager):
        expected_ = {"a": [REPLACEMENTS.get(el, el) for el in expected_["a"]]}

    df = nw.from_native(constructor_eager(data), eager_only=True)
    result_series = df["a"].str.to_titlecase()

    assert_equal_data({"a": result_series}, expected_)
