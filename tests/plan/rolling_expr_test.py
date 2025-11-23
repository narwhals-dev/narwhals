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
    return {"a": [None, 1, 2, None, 4, 6, 11]}


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
def test_rolling_sum_expr(
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
def test_rolling_mean_expr(
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
