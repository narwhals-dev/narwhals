from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, cast

import pytest

pytest.importorskip("numpy")
import numpy as np

import narwhals as nw
from tests.utils import assert_equal_series

if TYPE_CHECKING:
    from collections.abc import Sequence

    from narwhals._typing import EagerAllowed
    from narwhals.dtypes import NestedType
    from narwhals.typing import IntoDType, _1DArray


arr: _1DArray = cast("_1DArray", np.array([5, 2, 0, 1]))
NAME = "a"


def test_series_from_numpy(eager_backend: EagerAllowed) -> None:
    expected = [5, 2, 0, 1]
    result = nw.Series.from_numpy(NAME, arr, backend=eager_backend)
    assert isinstance(result, nw.Series)
    assert_equal_series(result, expected, NAME)


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (nw.Int16, [5, 2, 0, 1]),
        (nw.Int32(), [5, 2, 0, 1]),
        (nw.Float64, [5.0, 2.0, 0.0, 1.0]),
        (nw.Float32(), [5.0, 2.0, 0.0, 1.0]),
    ],
    ids=str,
)
def test_series_from_numpy_dtype(
    eager_backend: EagerAllowed, dtype: IntoDType, expected: Sequence[Any]
) -> None:
    result = nw.Series.from_numpy(NAME, arr, backend=eager_backend, dtype=dtype)
    assert result.dtype == dtype
    assert_equal_series(result, expected, NAME)


@pytest.mark.parametrize(
    ("bad_dtype", "message"),
    [
        (nw.List, r"nw.List.+not.+valid.+hint"),
        (nw.Struct, r"nw.Struct.+not.+valid.+hint"),
        (nw.Array, r"nw.Array.+not.+valid.+hint"),
        (np.floating, r"expected.+narwhals.+dtype.+floating"),
        (list[int], r"expected.+narwhals.+dtype.+(types.GenericAlias|list)"),
    ],
    ids=str,
)
def test_series_from_numpy_not_init_dtype(
    eager_backend: EagerAllowed, bad_dtype: type[NestedType] | object, message: str
) -> None:
    with pytest.raises(TypeError, match=re.compile(message, re.IGNORECASE | re.DOTALL)):
        nw.Series.from_numpy(NAME, arr, bad_dtype, backend=eager_backend)  # type: ignore[arg-type]


def test_series_from_numpy_not_eager() -> None:
    pytest.importorskip("ibis")
    with pytest.raises(ValueError, match="lazy-only"):
        nw.Series.from_numpy(NAME, arr, backend="ibis")  # type: ignore[arg-type]


def test_series_from_numpy_not_1d(eager_backend: EagerAllowed) -> None:
    with pytest.raises(ValueError, match="`from_numpy` only accepts 1D numpy arrays"):
        nw.Series.from_numpy(NAME, np.array([[0], [2]]), backend=eager_backend)  # pyright: ignore[reportArgumentType]
