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


# NOTE: Everything from here onwards is not supported on `main`


@pytest.fixture
def grouped() -> Data:
    return {
        "group": ["A", "A", "B", "B", "B"],
        "value_1": [1, 3, 3, 2, 3],
        "value_2": [1, 3, 3, 3, 3],
        "o_asc": [0, 1, 2, 3, 4],
        "o_null": [0, 1, 2, None, 4],
    }


GROUP = "group"
VALUE_1 = "value_1"
VALUE_2 = "value_2"
ORDER_ASC = "o_asc"
ORDER_NULL = "o_null"


# NOTE: For `pyarrow`, the result is identical to `order_by`, because the index is already in order
def test_is_first_last_distinct_partitioned(grouped: Data) -> None:
    expected = {
        GROUP: ["A", "A", "B", "B", "B"],
        VALUE_1: [1, 3, 3, 2, 3],
        "is_first_distinct": [True, True, True, True, False],
        "is_last_distinct": [True, True, False, True, True],
    }
    df = dataframe(grouped).drop(VALUE_2, ORDER_NULL)
    value = nwp.col(VALUE_1)
    result = (
        df.with_columns(
            is_first_distinct=value.is_first_distinct().over(GROUP),
            is_last_distinct=value.is_last_distinct().over(GROUP),
        )
        .sort(ORDER_ASC)
        .drop(ORDER_ASC)
    )
    assert_equal_data(result, expected)


# NOTE: This works the same as `polars`
def test_is_first_last_distinct_partitioned_order_by_desc(grouped: Data) -> None:
    expected = {
        GROUP: ["A", "A", "B", "B", "B"],
        VALUE_2: [1, 3, 3, 3, 3],
        # (1) Same result
        "first_distinct": [True, True, True, False, False],
        "last_distinct_desc": [True, True, True, False, False],
        # (2) Same result
        "last_distinct": [True, True, False, False, True],
        "first_distinct_desc": [True, True, False, False, True],
    }
    df = dataframe(grouped).drop(VALUE_1, ORDER_NULL)
    value = nwp.col(VALUE_2)
    first = value.is_first_distinct()
    last = value.is_last_distinct()

    result = (
        df.with_columns(
            first_distinct=first.over(GROUP, order_by=ORDER_ASC),
            last_distinct_desc=last.over(GROUP, order_by=ORDER_ASC, descending=True),
            last_distinct=last.over(GROUP, order_by=ORDER_ASC),
            first_distinct_desc=first.over(GROUP, order_by=ORDER_ASC, descending=True),
        )
        .sort(ORDER_ASC)
        .drop(ORDER_ASC)
    )
    assert_equal_data(result, expected)


# NOTE: `polars` *currently* ignores the `nulls_last` argument
# https://github.com/pola-rs/polars/issues/24989
def test_is_first_last_distinct_partitioned_order_by_nulls(grouped: Data) -> None:
    expected = {
        GROUP: ["A", "A", "B", "B", "B"],
        VALUE_2: [1, 3, 3, 3, 3],
        "first_distinct_nulls_first": [True, True, False, True, False],
        "last_distinct_nulls_first": [True, True, False, False, True],
        "first_distinct_nulls_last": [True, True, True, False, False],
        "last_distinct_nulls_last": [True, True, False, True, False],
    }
    df = dataframe(grouped).drop(VALUE_1)
    value = nwp.col(VALUE_2)
    first = value.is_first_distinct()
    last = value.is_last_distinct()
    result = (
        df.with_columns(
            first_distinct_nulls_first=first.over(GROUP, order_by=ORDER_NULL),
            last_distinct_nulls_first=last.over(GROUP, order_by=ORDER_NULL),
            first_distinct_nulls_last=first.over(
                GROUP, order_by=ORDER_NULL, nulls_last=True
            ),
            last_distinct_nulls_last=last.over(
                GROUP, order_by=ORDER_NULL, nulls_last=True
            ),
        )
        .sort(ORDER_ASC)
        .drop(ORDER_ASC, ORDER_NULL)
    )

    assert_equal_data(result, expected)
