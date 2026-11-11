from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw
from narwhals.exceptions import ShapeError
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


def test_scatter_scalar_broadcast(constructor_eager: ConstructorEager) -> None:
    s = series(constructor_eager, "a", [1, 2, 3, 4])
    assert_equal_series(s.scatter([0, 2], 999), [999, 2, 999, 4], "a")
    assert_equal_series(s.scatter([0, 2], [999]), [999, 2, 999, 4], "a")
    assert_equal_series(
        s.scatter([0, 2], series(constructor_eager, "v", [999])), [999, 2, 999, 4], "a"
    )


def test_scatter_broadcast_nested(
    constructor_eager: ConstructorEager, request: pytest.FixtureRequest
) -> None:
    if any(x in str(constructor_eager) for x in ("polars", "pyarrow_table")):
        reason = (
            "Neither Polars' scatter nor pyarrow's replace_with_mask support list types."
        )
        request.applymarker(pytest.mark.xfail(reason=reason))
    if "cudf" in str(constructor_eager):
        reason = "cuDF ListColumn._cast_setitem_value rejects Series values."
        request.applymarker(pytest.mark.xfail(reason=reason))
    s = series(constructor_eager, "a", [[1], [2], [3]])
    assert_equal_series(s.scatter([0, 2], [[9, 9]]), [[9, 9], [2], [9, 9]], "a")


def test_scatter_length_mismatch(
    constructor_eager: ConstructorEager, request: pytest.FixtureRequest
) -> None:
    if "polars" in str(constructor_eager):
        reason = (
            "Polars silently ignores extra values instead of raising "
            "(https://github.com/pola-rs/polars/issues/30713)."
        )
        request.applymarker(pytest.mark.xfail(reason=reason))
    s = series(constructor_eager, "a", [1, 2, 3, 4])
    with pytest.raises(ShapeError, match="Expected object of length 2"):
        s.scatter([0, 2], [7, 8, 9])


def test_scatter_casts_values(constructor_eager: ConstructorEager) -> None:
    s = series(constructor_eager, "a", [1.0, 2.0, 3.0])
    assert_equal_series(s.scatter([0, 2], 7), [7.0, 2.0, 7.0], "a")
    assert_equal_series(s.scatter([0, 2], [7, 8]), [7.0, 2.0, 8.0], "a")
    s = series(constructor_eager, "a", ["x", "y", "z"])
    assert_equal_series(s.scatter([0, 2], None), [None, "y", None], "a")


@pytest.mark.parametrize(
    ("indices", "values", "expected"),
    [
        ([0, 2], 99, [99, 2, 99, 4]),
        (0, 99, [99, 2, 3, 4]),
        ([0, 2], [99, 88], [99, 2, 88, 4]),
    ],
    ids=["broadcast", "single", "equal-length"],
)
def test_scatter_scalar_narrow_int_pyarrow(
    indices: int | list[int], values: int | list[int], expected: list[int]
) -> None:
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    s = nw.from_native(pa.chunked_array([[1, 2, 3, 4]], type=pa.int8()), series_only=True)
    output = s.scatter(indices, values)
    assert output.dtype == nw.Int8
    assert_equal_series(output, expected, "")


@pytest.mark.parametrize(
    ("indices", "values", "expected"),
    [
        ([-1, 0], [7, 8], [8, 2, 3, 7]),
        ([-4, -2], 9, [9, 2, 9, 4]),
        ([1, -1], [7, 8], [1, 7, 3, 8]),
    ],
    ids=["mixed", "broadcast", "single-negative"],
)
def test_scatter_negative_indices(
    indices: list[int],
    values: int | list[int],
    expected: list[int],
    constructor_eager: ConstructorEager,
) -> None:
    s = series(constructor_eager, "a", [1, 2, 3, 4])
    assert_equal_series(s.scatter(indices, values), expected, "a")


@pytest.mark.parametrize(
    ("indices", "values", "expected"),
    [
        ([0, 0, 2], [7, 8, 9], [8, 2, 9, 4]),
        ([2, 0, 2, 2], [7, 8, 9, 10], [8, 2, 10, 4]),
        ([-1, 3], [7, 8], [1, 2, 3, 8]),
        ([1, 1], 9, [1, 9, 3, 4]),
    ],
    ids=["repeated", "unordered", "negative-and-positive", "broadcast"],
)
def test_scatter_duplicate_indices(
    indices: list[int],
    values: int | list[int],
    expected: list[int],
    constructor_eager: ConstructorEager,
    request: pytest.FixtureRequest,
) -> None:
    if any(x in str(constructor_eager) for x in ("pandas_pyarrow", "modin_pyarrow")) and (
        not isinstance(values, int)
    ):
        reason = (
            "pandas' Arrow-backed setitem keeps the first value on duplicate indices "
            "(https://github.com/pandas-dev/pandas/issues/53996)."
        )
        request.applymarker(pytest.mark.xfail(reason=reason))
    s = series(constructor_eager, "a", [1, 2, 3, 4])
    assert_equal_series(s.scatter(indices, values), expected, "a")


@pytest.mark.parametrize("indices", [[0, 4], [-5], [10, 1]])
def test_scatter_out_of_bounds(
    indices: list[int], constructor_eager: ConstructorEager
) -> None:
    s = series(constructor_eager, "a", [1, 2, 3, 4])
    # Each backend raises its own exception type, so we can only match on the message.
    with pytest.raises(Exception, match="out of bounds"):
        s.scatter(indices, 9)


def test_scatter_null_indices(constructor_eager: ConstructorEager) -> None:
    s = series(constructor_eager, "a", [1, 2, 3, 4])
    # Polars and PyArrow raise ComputeError, pandas raises TypeError.
    with pytest.raises(Exception):  # noqa: B017, PT011
        s.scatter([0, None], [7, 8])  # type: ignore[list-item]
