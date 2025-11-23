from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals._plan as nwp
from tests.plan.utils import assert_equal_data, dataframe

if TYPE_CHECKING:
    from narwhals.typing import NonNestedLiteral
    from tests.conftest import Data

pytest.importorskip("pyarrow")


@pytest.fixture(scope="module")
def data() -> Data:
    return {
        "a": [None, 1, 2, None, 4, 6, 11],
        "b": [1, None, 2, None, 4, 6, 11],
        "c": [1, None, 2, 3, 4, 5, 6],
        "i": list(range(7)),
    }


@pytest.mark.parametrize(
    ("window_size", "min_samples", "center", "expected"),
    [
        (3, None, False, [None, None, None, None, None, None, 21]),
        (3, 1, False, [None, 1.0, 3.0, 3.0, 6.0, 10.0, 21.0]),
        (2, 1, False, [None, 1.0, 3.0, 2.0, 4.0, 10.0, 17.0]),
        (5, 1, True, [3.0, 3.0, 7.0, 13.0, 23.0, 21.0, 21.0]),
        (4, 1, True, [1.0, 3.0, 3.0, 7.0, 12.0, 21.0, 21.0]),
    ],
)
def test_rolling_sum(
    data: Data,
    window_size: int,
    *,
    min_samples: int | None,
    center: bool,
    expected: list[NonNestedLiteral],
) -> None:
    expr = nwp.col("a").rolling_sum(window_size, min_samples=min_samples, center=center)
    result = dataframe(data).select(expr)
    assert_equal_data(result, {"a": expected})


@pytest.mark.parametrize(
    ("window_size", "min_samples", "center", "expected"),
    [
        (3, None, False, [None, None, None, None, None, None, 7.0]),
        (3, 1, False, [None, 1.0, 1.5, 1.5, 3.0, 5.0, 7.0]),
        (2, 1, False, [None, 1.0, 1.5, 2.0, 4.0, 5.0, 8.5]),
        (5, 1, True, [1.5, 1.5, 7 / 3, 3.25, 5.75, 7.0, 7.0]),
        (4, 1, True, [1.0, 1.5, 1.5, 7 / 3, 4.0, 7.0, 7.0]),
    ],
)
def test_rolling_mean(
    data: Data,
    window_size: int,
    *,
    min_samples: int | None,
    center: bool,
    expected: list[NonNestedLiteral],
) -> None:
    expr = nwp.col("a").rolling_mean(window_size, min_samples=min_samples, center=center)
    result = dataframe(data).select(expr)
    assert_equal_data(result, {"a": expected})


@pytest.mark.parametrize(
    ("window_size", "min_samples", "center", "expected"),
    [
        (2, None, False, [None, None, 3, None, None, 10, 17]),
        (2, 2, False, [None, None, 3, None, None, 10, 17]),
        (3, 2, False, [None, None, 3, 3, 6, 10, 21]),
        (3, 1, False, [1, None, 3, 3, 6, 10, 21]),
        (3, 1, True, [3, 1, 3, 6, 10, 21, 17]),
        (4, 1, True, [3, 1, 3, 7, 12, 21, 21]),
        (5, 1, True, [3, 3, 7, 13, 23, 21, 21]),
    ],
)
def test_rolling_sum_order_by(
    data: Data,
    window_size: int,
    *,
    min_samples: int | None,
    center: bool,
    expected: list[NonNestedLiteral],
) -> None:
    expr = (
        nwp.col("b")
        .rolling_sum(window_size, min_samples=min_samples, center=center)
        .over(order_by="c")
    )
    result = dataframe(data).with_columns(expr).select("b", "i").sort("i").drop("i")
    assert_equal_data(result, {"b": expected})


@pytest.mark.parametrize(
    ("window_size", "min_samples", "center", "expected"),
    [
        (2, None, False, [None, None, 1.5, None, None, 5, 8.5]),
        (2, 2, False, [None, None, 1.5, None, None, 5, 8.5]),
        (3, 2, False, [None, None, 1.5, 1.5, 3, 5, 7]),
        (3, 1, False, [1, None, 1.5, 1.5, 3, 5, 7]),
        (3, 1, True, [1.5, 1, 1.5, 3, 5, 7, 8.5]),
        (4, 1, True, [1.5, 1, 1.5, 2.3333333333333335, 4, 7, 7]),
        (5, 1, True, [1.5, 1.5, 2.3333333333333335, 3.25, 5.75, 7.0, 7.0]),
    ],
)
def test_rolling_mean_order_by(
    data: Data,
    window_size: int,
    *,
    min_samples: int | None,
    center: bool,
    expected: list[NonNestedLiteral],
) -> None:
    expr = (
        nwp.col("b")
        .rolling_mean(window_size, min_samples=min_samples, center=center)
        .over(order_by="c")
    )
    result = dataframe(data).with_columns(expr).select("b", "i").sort("i").drop("i")
    assert_equal_data(result, {"b": expected})
