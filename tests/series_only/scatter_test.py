from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw
from tests.utils import ConstructorEager, assert_equal_data, assert_equal_series

if TYPE_CHECKING:
    from collections.abc import Collection


def series(frame: ConstructorEager, name: str, values: Collection[Any]) -> nw.Series[Any]:
    return nw.from_native(frame({name: values})).get_column(name)


@pytest.mark.filterwarnings(
    "ignore:.*all arguments of to_dict except for the argument:FutureWarning"
)
@pytest.mark.parametrize(
    ("data", "indices", "values", "expected"),
    [
        ([142, 124, 13], [0, 2, 1], (142, 124, 13), [142, 13, 124]),
        ([1, 2, 3], 0, 999, [999, 2, 3]),
        (
            [16, 12, 10, 9, 6, 5, 2],
            (6, 1, 0, 5, 3, 2, 4),
            [16, 12, 10, 9, 6, 5, 2],
            [10, 12, 5, 6, 2, 9, 16],
        ),
        ([5.5, 9.2, 1.0], (), (), [5.5, 9.2, 1.0]),
    ],
    ids=["single-series", "integer", "unordered-indices", "empty-indices"],
)
def test_scatter(
    data: list[Any],
    indices: int | Collection[int],
    values: int | Collection[int],
    expected: list[Any],
    constructor_eager: ConstructorEager,
) -> None:
    constructor = partial(series, constructor_eager)
    s = constructor("s", data)
    df = s.to_frame().with_row_index("dont change me")
    unchanged_indexed = df.to_dict(as_series=False)
    assert_equal_series(s.scatter(indices, values), expected, "s")
    if not isinstance(indices, int):
        assert_equal_series(s.scatter(constructor("i", indices), values), expected, "s")
    if not isinstance(values, int):
        assert_equal_series(s.scatter(indices, constructor("v", values)), expected, "s")
    assert_equal_data(df, unchanged_indexed)


def test_scatter_pandas_index() -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    s = nw.from_native(pd.Series([2, 3, 6], index=[1, 0, 2]), series_only=True)
    result = s.scatter([1, 0, 2], s)
    expected = pd.Series([3, 2, 6], index=[1, 0, 2])
    pd.testing.assert_series_equal(result.to_native(), expected)


def test_scatter_2862(constructor_eager: ConstructorEager) -> None:
    s = series(constructor_eager, "a", [1, 2, 3])
    assert_equal_series(s.scatter(1, 999), [1, 999, 3], "a")
    assert_equal_series(s.scatter([0, 2], [999, 888]), [999, 2, 888], "a")
    assert_equal_series(s.scatter([2, 0], [999, 888]), [888, 2, 999], "a")
