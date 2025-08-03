from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw
from tests.utils import assert_equal_series

if TYPE_CHECKING:
    from narwhals._namespace import EagerAllowed


def test_series_from_iterable(eager_backend: EagerAllowed) -> None:
    name = "b"
    into_values = 4, 1, 2
    values = list(into_values)
    result = nw.Series.from_iterable(name, values, backend=eager_backend)
    # all supported libraries auto-infer this to be int64, we can always special-case
    # something different if necessary
    assert result.dtype == nw.Int64
    assert_equal_series(result, values, name)

    result = nw.Series.from_iterable(name, values, nw.Int32, backend=eager_backend)
    assert result.dtype == nw.Int32
    assert_equal_series(result, values, name)


def test_series_from_iterable_not_eager() -> None:
    backend = "sqlframe"
    pytest.importorskip(backend)
    with pytest.raises(ValueError, match="lazy-only"):
        nw.Series.from_iterable("", [1, 2, 3], backend=backend)


def test_series_from_iterable_numpy_not_1d(eager_backend: EagerAllowed) -> None:
    pytest.importorskip("numpy")
    import numpy as np

    with pytest.raises(ValueError, match="only.+1D numpy arrays"):
        nw.Series.from_iterable("", np.array([[0], [2]]), backend=eager_backend)


def test_series_from_iterable_not_iterable(eager_backend: EagerAllowed) -> None:
    with pytest.raises(TypeError, match="iterable.+got.+int"):
        nw.Series.from_iterable("", 2000, backend=eager_backend)  # type: ignore[arg-type]
