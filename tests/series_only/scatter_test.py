from __future__ import annotations

from datetime import timedelta
from functools import partial
from typing import TYPE_CHECKING, Any

import pytest
from packaging.version import Version as PkgVersion

import narwhals as nw
from narwhals.exceptions import ComputeError, ShapeError
from tests.utils import (
    PANDAS_VERSION,
    PYARROW_VERSION,
    ConstructorEager,
    assert_equal_data,
    assert_equal_series,
)

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
    if "polars" in str(constructor_eager):
        reason = "Polars' scatter doesn't support list types."
        request.applymarker(pytest.mark.xfail(reason=reason))
    if "cudf" in str(constructor_eager):
        reason = "cuDF ListColumn._cast_setitem_value rejects Series values."
        request.applymarker(pytest.mark.xfail(reason=reason))
    s = series(constructor_eager, "a", [[1], [2], [3]])
    assert_equal_series(s.scatter([0, 2], [[9, 9]]), [[9, 9], [2], [9, 9]], "a")


def test_scatter_length_mismatch(constructor_eager: ConstructorEager) -> None:
    s = series(constructor_eager, "a", [1, 2, 3, 4])
    with pytest.raises(ShapeError, match="Expected object of length 2"):
        s.scatter([0, 2], [7, 8, 9])


@pytest.mark.parametrize(
    ("indices", "values", "expected"),
    [
        ([-1], [9], [1, 2, 3, 9]),
        ([-1, 0], [9, 8], [8, 2, 3, 9]),
        ([-4, 1], 7, [7, 7, 3, 4]),
        ([0, 0, 2], [10, 20, 30], [20, 2, 30, 4]),
        ([2, 0, 2, -4], [10, 20, 30, 40], [40, 2, 30, 4]),
        ([1, 1], 5, [1, 5, 3, 4]),
    ],
    ids=[
        "negative",
        "negative-and-positive",
        "negative-broadcast",
        "duplicates-last-wins",
        "duplicates-unordered-with-negative",
        "duplicates-broadcast",
    ],
)
def test_scatter_negative_and_duplicate_indices(
    indices: list[int],
    values: int | list[int],
    expected: list[int],
    constructor_eager: ConstructorEager,
) -> None:
    s = series(constructor_eager, "a", [1, 2, 3, 4])
    assert_equal_series(s.scatter(indices, values), expected, "a")


@pytest.mark.parametrize("indices", [[4], [-5], [0, 10]])
def test_scatter_out_of_bounds(
    indices: list[int], constructor_eager: ConstructorEager
) -> None:
    s = series(constructor_eager, "a", [1, 2, 3, 4])
    # Error type is backend-specific (e.g. `IndexError`, Polars' `OutOfBoundsError`).
    with pytest.raises(Exception, match=r"(?i)out of bounds"):
        s.scatter(indices, 0)


@pytest.mark.parametrize("dtype", [nw.Int8, nw.UInt8])
def test_scatter_narrow_index_dtype(
    dtype: type[nw.Int8 | nw.UInt8], constructor_eager: ConstructorEager
) -> None:
    s = series(constructor_eager, "a", list(range(200)))
    indices = series(constructor_eager, "i", [0, 1, 0]).cast(dtype)
    result = s.scatter(indices, series(constructor_eager, "v", [7, 8, 9]))
    assert_equal_series(result.head(3), [9, 8, 2], "a")


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


def test_scatter_null_indices(constructor_eager: ConstructorEager) -> None:
    s = series(constructor_eager, "a", [1, 2, 3, 4])
    indices = series(constructor_eager, "i", [0, None])
    with pytest.raises(ComputeError, match="should not be null"):
        s.scatter(indices, [7, 8])


@pytest.mark.parametrize("dtype", [nw.Int8, nw.Int32, nw.UInt8])
def test_scatter_keeps_narrow_int_dtype(
    dtype: type[nw.Int8 | nw.Int32 | nw.UInt8], constructor_eager: ConstructorEager
) -> None:
    s = series(constructor_eager, "a", [1, 2, 3]).cast(dtype)
    result = s.scatter([0, 1], [7, 8])
    assert result.dtype == dtype
    assert_equal_series(result, [7, 8, 3], "a")
    result = s.scatter([0, 2], 9)
    assert result.dtype == dtype
    assert_equal_series(result, [9, 2, 9], "a")


@pytest.mark.skipif(
    PANDAS_VERSION < PkgVersion("3"), reason="pandas<3 upcasts the column"
)
def test_scatter_int_overflow_pandas() -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    s = nw.from_native(pd.Series([1, 2, 3], dtype="int8"), series_only=True)
    # Out-of-range values must not be wrapped around by the cast to the column dtype.
    with pytest.raises(TypeError, match="Invalid value"):
        s.scatter([0], [300])


@pytest.mark.parametrize(
    ("data", "pa_type", "value", "expected"),
    [
        pytest.param(
            ["a", "b", "a"],
            "dictionary",
            "b",
            ["b", "b", "b"],
            marks=pytest.mark.skipif(
                PYARROW_VERSION < PkgVersion("15"),
                reason="pyarrow<15 can't cast string to dictionary",
            ),
        ),
        ([timedelta(1), timedelta(1), timedelta(3)], "duration", timedelta(2), None),
        ([{"x": 1}, {"x": 2}, {"x": 3}], "struct", {"x": 9}, None),
    ],
    ids=["dictionary", "duration", "struct"],
)
def test_scatter_pyarrow_types_without_replace_with_mask(
    data: list[Any], pa_type: str, value: Any, expected: list[Any] | None
) -> None:
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    arr = pa.array(data)
    if pa_type == "dictionary":
        arr = arr.dictionary_encode()
    s = nw.from_native(pa.chunked_array([arr[:2], arr[2:]]), series_only=True)
    result = s.scatter([2, 0], [value, value])
    assert result.dtype == s.dtype
    if expected is None:
        expected = [value, data[1], value]
    assert result.to_list() == expected
