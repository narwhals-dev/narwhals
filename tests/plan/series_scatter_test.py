from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from tests.plan.utils import DataFrame, Series, assert_equal_data, assert_equal_series

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


@pytest.mark.parametrize(
    ("data", "indices", "values", "expected", "values_into_series"),
    [
        ([1, 2, 3], [0, 1], [999, 888], [999, 888, 3], False),
        ([142, 124, 13], [0, 2, 1], [142, 124, 13], [142, 13, 124], True),
        ([1, 2, 3], 0, 999, [999, 2, 3], False),
        (
            [16, 12, 10, 9, 6, 5, 2],
            [6, 1, 0, 5, 3, 2, 4],
            [16, 12, 10, 9, 6, 5, 2],
            [10, 12, 5, 6, 2, 9, 16],
            True,
        ),
        ([5.5, 9.2, 1.0], (), (), [5.5, 9.2, 1.0], False),
    ],
    ids=["lists", "single-series", "integer", "unordered-indices", "empty-indices"],
)
def test_scatter(
    series: Series,
    data: list[Any],
    indices: int | Sequence[int],
    values: Iterable[int],
    expected: list[Any],
    values_into_series: bool,  # noqa: FBT001
) -> None:
    if values_into_series:
        values = series(values)
    result = series(data).alias("ser").scatter(indices, values)
    assert_equal_series(result, expected, "ser")


def test_scatter_unchanged(dataframe: DataFrame) -> None:
    df = dataframe({"a": [1, 2, 3], "b": [142, 124, 132]})
    a = df.get_column("a")
    b = df.get_column("b")
    a_scatter, b_scatter = (
        a.scatter([0, 1], [999, 888]),
        b.scatter([0, 2, 1], [142, 124, 132]),
    )
    df.with_columns(a_scatter, b_scatter)
    assert_equal_data(df, {"a": [1, 2, 3], "b": [142, 124, 132]})


def test_scatter_2862(series: Series) -> None:
    ser = series([1, 2, 3]).alias("a")
    assert_equal_series(ser.scatter(1, 999), [1, 999, 3], "a")
    assert_equal_series(ser.scatter([0, 2], [999, 888]), [999, 2, 888], "a")
    assert_equal_series(ser.scatter([2, 0], [999, 888]), [888, 2, 999], "a")
