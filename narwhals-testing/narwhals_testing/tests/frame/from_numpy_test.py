from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

pytest.importorskip("numpy")
import numpy as np

import narwhals as nw
from tests.utils import assert_equal_data

if TYPE_CHECKING:
    from narwhals._typing import EagerAllowed
    from narwhals.typing import _2DArray


arr: _2DArray = cast("_2DArray", np.array([[5, 2, 0, 1], [1, 4, 7, 8], [1, 2, 3, 9]]))
expected = {
    "column_0": [5, 1, 1],
    "column_1": [2, 4, 2],
    "column_2": [0, 7, 3],
    "column_3": [1, 8, 9],
}


def test_from_numpy(eager_backend: EagerAllowed) -> None:
    result = nw.DataFrame.from_numpy(arr, backend=eager_backend)
    assert_equal_data(result, expected)
    assert isinstance(result, nw.DataFrame)


def test_from_numpy_schema_dict(eager_backend: EagerAllowed) -> None:
    schema = {"c": nw.Int16(), "d": nw.Float32(), "e": nw.Int16(), "f": nw.Float64()}
    result = nw.DataFrame.from_numpy(arr, backend=eager_backend, schema=schema)
    assert result.collect_schema() == schema


def test_from_numpy_schema_list(eager_backend: EagerAllowed) -> None:
    schema = ["c", "d", "e", "f"]
    result = nw.DataFrame.from_numpy(arr, backend=eager_backend, schema=schema)
    assert result.columns == schema


def test_from_numpy_schema_notvalid(eager_backend: EagerAllowed) -> None:
    with pytest.raises(TypeError, match=r"`schema.*expected.*types"):
        nw.DataFrame.from_numpy(arr, schema=5, backend=eager_backend)  # type: ignore[arg-type]


def test_from_numpy_non_eager() -> None:
    pytest.importorskip("duckdb")
    with pytest.raises(ValueError, match="lazy-only"):
        nw.DataFrame.from_numpy(arr, backend="duckdb")  # type: ignore[arg-type]


def test_from_numpy_not2d(eager_backend: EagerAllowed) -> None:
    with pytest.raises(ValueError, match="`from_numpy` only accepts 2D numpy arrays"):
        nw.DataFrame.from_numpy(np.array([0]), backend=eager_backend)  # pyright: ignore[reportArgumentType]
