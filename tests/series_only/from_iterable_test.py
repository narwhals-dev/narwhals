from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw
from tests.utils import assert_equal_series

if TYPE_CHECKING:
    from collections.abc import Sequence

    from narwhals._namespace import EagerAllowed
    from narwhals.typing import IntoDType


# TODO @dangotbanned: parametrize many https://github.com/narwhals-dev/narwhals/pull/2933#discussion_r2249983202
def test_series_from_iterable(eager_backend: EagerAllowed) -> None:
    name = "b"
    into_values = 4, 1, 2
    values = list(into_values)
    result = nw.Series.from_iterable(name, values, nw.Int32, backend=eager_backend)
    assert result.dtype == nw.Int32
    assert_equal_series(result, values, name)


@pytest.mark.parametrize(("values", "expected_dtype"), [((4, 1, 2), nw.Int64)])
def test_series_from_iterable_infer(
    eager_backend: EagerAllowed, values: Sequence[Any], expected_dtype: IntoDType
) -> None:
    name = "b"
    result = nw.Series.from_iterable(name, values, backend=eager_backend)
    assert result.dtype == expected_dtype
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
