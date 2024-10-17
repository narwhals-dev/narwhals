from __future__ import annotations

from typing import TYPE_CHECKING

import narwhals.stable.v1 as nw
from narwhals.dependencies import get_cudf
from narwhals.dependencies import get_numpy

if TYPE_CHECKING:
    from tests.utils import ConstructorEager


def test_to_py_scalar(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager({"a": [1, 2, 3]}))
    assert nw.to_py_scalar(df["a"].item(0)) == 1


def test_to_py_scalar_arrays_series() -> None:
    if cudf := get_cudf():  # pragma: no cover
        df = nw.from_native(cudf.DataFrame({"a": [1, 2, 3]}))
        cudf_series = nw.to_native(nw.to_py_scalar(df["a"]))
        assert isinstance(cudf_series, cudf.Series)

    np = get_numpy()
    array = np.array([1, 2, 3])
    assert isinstance(nw.to_py_scalar(array), np.ndarray)
