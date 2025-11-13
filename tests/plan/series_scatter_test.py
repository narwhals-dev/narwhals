from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import narwhals._plan as nwp
from tests.plan.utils import assert_equal_data, assert_equal_series, dataframe, series

if TYPE_CHECKING:
    from collections.abc import Sequence

    from narwhals._plan.typing import OneOrIterable


XFAIL_NOT_IMPL_SCATTER = pytest.mark.xfail(
    reason="`Series.scatter` is not yet implemented", raises=NotImplementedError
)


@pytest.mark.parametrize(
    ("data", "indices", "values", "expected"),
    [
        pytest.param([1, 2, 3], [0, 1], [999, 888], [999, 888, 3]),
        pytest.param(
            [142, 124, 13],
            [0, 2, 1],
            series([142, 124, 13]),
            [142, 132, 124],
            marks=pytest.mark.xfail(
                reason=(
                    "BUG:"  # no: fmt
                    "Expected: {'ser': [142, 132, 124]}\n]"  # no: fmt
                    "Got: {'ser': [142, 13, 124]}"
                ),
                raises=AssertionError,
            ),
        ),
        pytest.param([1, 2, 3], 0, 999, [999, 2, 3]),
        pytest.param(
            [16, 12, 10, 9, 6, 5, 2],
            [6, 1, 0, 5, 3, 2, 4],
            series([16, 12, 10, 9, 6, 5, 2]),
            [10, 12, 5, 6, 2, 9, 16],
        ),
        pytest.param([5.5, 9.2, 1.0], (), (), [5.5, 9.2, 1.0]),
    ],
    ids=["lists", "single-series", "integer", "unordered-indices", "empty-indices"],
)
def test_scatter(
    data: list[Any],
    indices: int | Sequence[int],
    values: OneOrIterable[int],
    expected: list[Any],
) -> None:
    ser = series(data).alias("ser")
    if isinstance(values, nwp.Series):
        assert ser.implementation is values.implementation
    assert_equal_series(ser.scatter(indices, values), expected, "ser")


def test_scatter_unchanged() -> None:
    df = dataframe({"a": [1, 2, 3], "b": [142, 124, 132]})
    a = df.get_column("a")
    b = df.get_column("b")
    df.with_columns(a.scatter([0, 1], [999, 888]), b.scatter([0, 2, 1], [142, 124, 132]))
    assert_equal_data(df, {"a": [1, 2, 3], "b": [142, 124, 132]})


def test_scatter_2862() -> None:
    ser = series([1, 2, 3]).alias("a")
    assert_equal_series(ser.scatter(1, 999), [1, 999, 3], "a")
    assert_equal_series(ser.scatter([0, 2], [999, 888]), [999, 2, 888], "a")
    assert_equal_series(ser.scatter([2, 0], [999, 888]), [888, 2, 999], "a")
