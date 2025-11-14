from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from narwhals import _plan as nwp
from tests.plan.utils import assert_equal_data, dataframe

if TYPE_CHECKING:
    from tests.conftest import Data


@pytest.fixture
def data() -> Data:
    return {"a": [1, 1, 2, 3, 3, 2], "b": [1, 2, 3, 2, 1, 3]}


@pytest.fixture
def data_indexed(data: Data) -> Data:
    return data | {"i": [None, 1, 2, 3, 4, 5]}


@pytest.fixture
def expected() -> Data:
    return {
        "a": [True, False, True, True, False, False],
        "b": [True, True, True, False, False, False],
    }


@pytest.fixture
def expected_invert(expected: Data) -> Data:
    return {k: [not el for el in v] for k, v in expected.items()}


# NOTE: Isn't supported on `main` for `pyarrow` + lots of other cases (non-elementary group-by agg)
# Could be interesting to attempt here?
XFAIL_PARTITIONED_AND_ORDERED_DISTINCT = pytest.mark.xfail(
    reason="TODO: Support ordereding as well!", raises=AssertionError
)


def test_is_first_distinct(data: Data, expected: Data) -> None:
    result = dataframe(data).select(nwp.all().is_first_distinct())
    assert_equal_data(result, expected)


def test_is_last_distinct(data: Data, expected_invert: Data) -> None:
    result = dataframe(data).select(nwp.all().is_last_distinct())
    assert_equal_data(result, expected_invert)


def test_is_first_distinct_order_by(data_indexed: Data, expected: Data) -> None:
    result = (
        dataframe(data_indexed)
        .select(nwp.col("a", "b").is_first_distinct().over(order_by="i"), "i")
        .sort("i")
        .drop("i")
    )
    assert_equal_data(result, expected)


def test_is_last_distinct_order_by(data_indexed: Data, expected_invert: Data) -> None:
    result = (
        dataframe(data_indexed)
        .select(nwp.col("a", "b").is_last_distinct().over(order_by="i"), "i")
        .sort("i")
        .drop("i")
    )
    assert_equal_data(result, expected_invert)


@pytest.fixture
def grouped() -> Data:
    return {
        "group": ["A", "A", "B", "B", "B"],
        "value": [1, 3, 3, 2, 3],
        "o_asc": [0, 1, 2, 3, 4],
    }


# NOTE: For `pyarrow`, the result is identical to `order_by`, because the index is already in order
def test_is_first_last_distinct_partitioned(grouped: Data) -> None:
    expected = {
        "group": ["A", "A", "B", "B", "B"],
        "value": [1, 3, 3, 2, 3],
        "is_first_distinct": [True, True, True, True, False],
        "is_last_distinct": [True, True, False, True, True],
    }

    result = (
        dataframe(grouped)
        .with_columns(
            is_first_distinct=nwp.col("value").is_first_distinct().over("group"),
            is_last_distinct=nwp.col("value").is_last_distinct().over("group"),
        )
        .sort("o_asc")
        .drop("o_asc")
    )
    assert_equal_data(result, expected)


@XFAIL_PARTITIONED_AND_ORDERED_DISTINCT
def test_is_first_distinct_partitioned_order_by(grouped: Data) -> None:
    expected = {
        "group": ["A", "A", "B", "B", "B"],
        "value": [1, 3, 3, 2, 3],
        "is_first_distinct_desc": [True, True, False, True, True],
        "is_last_distinct_desc": [True, True, True, True, False],
    }
    value = nwp.col("value")
    result = (
        dataframe(grouped)
        .with_columns(
            is_first_distinct_desc=value.is_first_distinct().over(
                "group", order_by="o_asc", descending=True
            ),
            is_last_distinct_desc=value.is_last_distinct().over(
                "group", order_by="o_asc", descending=True
            ),
        )
        .sort("o_asc")
        .drop("o_asc")
    )
    assert_equal_data(result, expected)


@XFAIL_PARTITIONED_AND_ORDERED_DISTINCT
def test_is_last_distinct_partitioned_order_by_nulls() -> None:
    data_ = {
        "group": ["A", "A", "B", "B", "B"],
        "value": [1, 3, 3, 3, 3],
        "o_asc": [0, 1, 2, 3, 4],
        "o_null": [0, 1, 2, None, 4],
    }
    df = dataframe(data_)
    expected = {
        "group": ["A", "A", "B", "B", "B"],
        "value": [1, 3, 3, 3, 3],
        "first_distinct_nulls_first": [True, True, False, True, False],
        "first_distinct_nulls_last": [True, True, True, False, False],
        "last_distinct_nulls_first": [True, True, False, False, True],
        "last_distinct_nulls_last": [True, True, False, True, False],
    }
    value = nwp.col("value")
    first = value.is_first_distinct()
    last = value.is_last_distinct()
    result = (
        df.with_columns(
            first.over("group", order_by="o_null").alias("first_distinct_nulls_first"),
            first.over("group", order_by="o_null", nulls_last=True).alias(
                "first_distinct_nulls_last"
            ),
            last.over("group", order_by="o_null").alias("last_distinct_nulls_first"),
            last.over("group", order_by="o_null", nulls_last=True).alias(
                "last_distinct_nulls_last"
            ),
        )
        .sort("o_asc")
        .drop("o_asc", "o_null")
    )

    assert_equal_data(result, expected)
