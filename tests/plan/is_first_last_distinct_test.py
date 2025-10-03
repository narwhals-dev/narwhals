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
def data_alt_1() -> Data:
    return {"a": [1, 1, 2, 2, 2], "b": [1, 3, 3, 2, 3]}


@pytest.fixture
def data_alt_1_indexed(data_alt_1: Data) -> Data:
    return data_alt_1 | {"i": [0, 1, 2, 3, 4]}


@pytest.fixture
def data_alt_2() -> Data:
    return {"a": [1, 1, 2, 2, 2], "b": [1, 2, 2, 2, 1]}


@pytest.fixture
def data_alt_2_indexed(data_alt_2: Data) -> Data:
    return data_alt_2 | {"i": [None, 1, 2, 3, 4]}


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
XFAIL_PARTITIONED_ORDER_BY = pytest.mark.xfail(
    reason="Not supporting `over(*partition_by, order_by=...)` yet",
    raises=NotImplementedError,
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


@XFAIL_PARTITIONED_ORDER_BY
def test_is_first_distinct_partitioned_order_by(
    data_alt_1_indexed: Data,
) -> None:  # pragma: no cover
    expected = {"b": [True, True, True, True, False]}
    result = (
        dataframe(data_alt_1_indexed)
        .select(nwp.col("b").is_first_distinct().over("a", order_by="i"), "i")
        .sort("i")
        .drop("i")
    )
    assert_equal_data(result, expected)


@XFAIL_PARTITIONED_ORDER_BY
def test_is_last_distinct_partitioned_order_by(
    data_alt_1_indexed: Data,
) -> None:  # pragma: no cover
    expected = {"b": [True, True, False, True, True]}
    result = (
        dataframe(data_alt_1_indexed)
        .select(nwp.col("b").is_last_distinct().over("a", order_by="i"), "i")
        .sort("i")
        .drop("i")
    )
    assert_equal_data(result, expected)


@XFAIL_PARTITIONED_ORDER_BY
def test_is_last_distinct_partitioned_order_by_nulls(
    data_alt_2_indexed: Data,
) -> None:  # pragma: no cover
    expected = {"b": [True, True, False, True, True]}
    result = (
        dataframe(data_alt_2_indexed)
        .select(nwp.col("b").is_last_distinct().over("a", order_by="i"), "i")
        .sort("i")
        .drop("i")
    )
    assert_equal_data(result, expected)
