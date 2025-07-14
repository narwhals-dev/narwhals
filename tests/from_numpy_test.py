from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

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
        nw.from_numpy(arr, backend="duckdb")


def test_from_numpy_not2d(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data))
    backend = nw.get_native_namespace(df)
    with pytest.raises(ValueError, match="`from_numpy` only accepts 2D numpy arrays"):
        nw.from_numpy(np.array([0]), backend=backend)  # pyright: ignore[reportArgumentType]
