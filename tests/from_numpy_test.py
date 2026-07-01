from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

pytest.importorskip("numpy")
import numpy as np

import narwhals as nw
from tests.utils import ConstructorEager, assert_equal_data

if TYPE_CHECKING:
    from narwhals.typing import _2DArray

data = {"a": [1, 2, 3], "b": [4, 5, 6]}
arr: _2DArray = cast("_2DArray", np.array([[5, 2, 0, 1], [1, 4, 7, 8], [1, 2, 3, 9]]))
expected = {
    "column_0": [5, 1, 1],
    "column_1": [2, 4, 2],
    "column_2": [0, 7, 3],
    "column_3": [1, 8, 9],
}


def test_from_numpy(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data))
    backend = nw.get_native_namespace(df)
    result = nw.from_numpy(arr, backend=backend)
    assert_equal_data(result, expected)
    assert isinstance(result, nw.DataFrame)


def test_from_numpy_schema_dict(constructor_eager: ConstructorEager) -> None:
    schema = {"c": nw.Int16(), "d": nw.Float32(), "e": nw.Int16(), "f": nw.Float64()}
    df = nw.from_native(constructor_eager(data))
    backend = nw.get_native_namespace(df)
    result = nw.from_numpy(arr, backend=backend, schema=schema)
    assert result.collect_schema() == schema


def test_from_numpy_schema_list(constructor_eager: ConstructorEager) -> None:
    schema = ["c", "d", "e", "f"]
    df = nw.from_native(constructor_eager(data))
    backend = nw.get_native_namespace(df)
    result = nw.from_numpy(arr, backend=backend, schema=schema)
    assert result.columns == schema


def test_from_numpy_schema_notvalid(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data))
    backend = nw.get_native_namespace(df)
    with pytest.raises(TypeError, match=r"`schema.*expected.*types"):
        nw.from_numpy(arr, schema=5, backend=backend)  # type: ignore[arg-type]


def test_from_numpy_non_eager() -> None:
    pytest.importorskip("duckdb")
    with pytest.raises(ValueError, match="lazy-only"):
        nw.from_numpy(arr, backend="duckdb")  # type: ignore[arg-type]


def test_from_numpy_not2d(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data))
    backend = nw.get_native_namespace(df)
    with pytest.raises(ValueError, match="`from_numpy` only accepts 2D numpy arrays"):
        nw.from_numpy(np.array([0]), backend=backend)  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize("schema", [None, ["x", "y", "z"]])
def test_from_numpy_square(
    constructor_eager: ConstructorEager, schema: list[str] | None
) -> None:
    # See https://github.com/narwhals-dev/narwhals/issues/3716:
    # Fortran-contiguous square array (as returned by polars' `to_numpy`) used to be
    # silently transposed on the polars backend, since polars infers column
    # orientation when the schema length matches both axes.
    rows = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    columns = [f"column_{i}" for i in range(3)] if schema is None else schema
    expected = {name: values for name, *values in zip(columns, *rows, strict=True)}

    square_arr = np.asfortranarray(rows)
    backend = nw.get_native_namespace(
        nw.from_native(constructor_eager(data), eager_only=True)
    )

    result = nw.from_numpy(square_arr, backend=backend, schema=schema)
    assert_equal_data(result, expected)


def test_from_numpy_square_roundtrip(constructor_eager: ConstructorEager) -> None:
    square_data = {"a": [0, 3, 6], "b": [1, 4, 7], "c": [2, 5, 8]}
    df = nw.from_native(constructor_eager(square_data), eager_only=True)
    backend = nw.get_native_namespace(df)
    result = nw.from_numpy(df.to_numpy(), backend=backend, schema=df.columns)
    assert_equal_data(result, square_data)
