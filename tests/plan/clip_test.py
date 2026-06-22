from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from narwhals import _plan as nwp
from narwhals.exceptions import MultiOutputExpressionError
from tests.plan.utils import DataFrame, assert_equal_data

if TYPE_CHECKING:
    from narwhals._plan.typing import IntoExprColumn
    from narwhals.typing import NumericLiteral, TemporalLiteral


@pytest.mark.parametrize(
    ("lower", "upper", "expected"),
    [
        (3, 4, [3, 3, 3, 3, 4]),
        (0, 4, [1, 2, 3, 0, 4]),
        (None, 4, [1, 2, 3, -4, 4]),
        (-2, 0, [0, 0, 0, -2, 0]),
        (-2, None, [1, 2, 3, -2, 5]),
        ("lb", nwp.col("ub") + 1, [3, 2, 3, 1, 3]),
    ],
)
def test_clip_expr(
    lower: IntoExprColumn | NumericLiteral | TemporalLiteral | None,
    upper: IntoExprColumn | NumericLiteral | TemporalLiteral | None,
    expected: list[int],
    dataframe: DataFrame,
) -> None:
    data = {"a": [1, 2, 3, -4, 5], "lb": [3, 2, 1, 1, 1], "ub": [4, 4, 2, 2, 2]}
    result = dataframe(data).select(nwp.col("a").clip(lower, upper))
    assert_equal_data(result, {"a": expected})


def test_clip_expr_series(dataframe: DataFrame) -> None:
    expected = [1, 2, 3, 4, 5]
    data = {"a": [1, 2, 3, -4, 5], "lb": [3, 2, 1, 1, 1], "ub": [4, 4, 2, 2, 2]}
    df = dataframe(data)
    lower = df.to_series().from_iterable(
        [1, 1, 2, 4, 3], backend=dataframe.implementation
    )
    result = dataframe(data).select(nwp.col("a").clip(lower))
    assert_equal_data(result, {"a": expected})


def test_clip_invalid(dataframe: DataFrame) -> None:
    df = dataframe({"a": [1, 2, 3], "b": [4, 5, 6]})
    with pytest.raises(MultiOutputExpressionError):
        df.select(nwp.col("a").clip(nwp.all(), nwp.col("a", "b")))
