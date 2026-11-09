from __future__ import annotations

from typing import Any

import pytest
from packaging.version import Version as PkgVersion

pytest.importorskip("numpy")
import numpy as np

import narwhals as nw
from tests.utils import NUMPY_VERSION, ConstructorEager

pytestmark = pytest.mark.skipif(
    NUMPY_VERSION < PkgVersion("1.23"), reason="`np.from_dlpack` needs numpy>=1.23"
)


class LegacyConsumerView:
    """Expose only the unversioned DLPack signature, like pre-1.0 consumers do."""

    def __init__(self, obj: Any) -> None:
        self._obj = obj

    def __dlpack__(self, *, stream: Any = None) -> Any:
        return self._obj.__dlpack__(stream=stream)


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ({"a": [1, 2, 3], "b": [4, 5, 6]}, np.array([[1, 4], [2, 5], [3, 6]])),
        ({"a": [1, 2, 3], "b": [4.5, 5.0, 6.0]}, np.array([[1, 4.5], [2, 5], [3, 6]])),
        (
            {"a": [True, False, True], "b": [False, False, True]},
            np.array([[True, False], [False, False], [True, True]]),
        ),
        ({"a": [1, None, 3], "b": [4, 5, 6]}, np.array([[1, 4], [np.nan, 5], [3, 6]])),
    ],
)
def test_dlpack(
    request: pytest.FixtureRequest,
    constructor_eager: ConstructorEager,
    data: dict[str, Any],
    expected: np.ndarray[Any, Any],
) -> None:
    if expected.dtype == bool and NUMPY_VERSION < PkgVersion("1.25"):  # pragma: no cover
        # numpy<1.25 can't export booleans via DLPack.
        request.applymarker(pytest.mark.xfail(raises=BufferError))
    if expected.dtype == bool and any(
        name in str(constructor_eager) for name in ("nullable", "_pyarrow_constructor")
    ):
        # `DataFrame.to_numpy` returns `object` for pandas `boolean` and `bool[pyarrow]`
        # columns, unlike `Series.to_numpy`.
        request.applymarker(pytest.mark.xfail(raises=BufferError))
    df = nw.from_native(constructor_eager(data), eager_only=True)
    assert df.__dlpack_device__() == (1, 0)
    result = np.from_dlpack(df)
    assert result.dtype == expected.dtype
    np.testing.assert_array_equal(result, expected)


def test_dlpack_legacy_consumer(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager({"a": [1, 2], "b": [3, 4]}), eager_only=True)
    result = np.from_dlpack(LegacyConsumerView(df))
    np.testing.assert_array_equal(result, np.array([[1, 3], [2, 4]]))


def test_dlpack_unsupported_dtype(constructor_eager: ConstructorEager) -> None:
    data = {"a": [1, 2], "b": ["x", "y"]}
    df = nw.from_native(constructor_eager(data), eager_only=True)
    with pytest.raises(BufferError):
        np.from_dlpack(df)


def test_dlpack_defers_to_native(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    received: dict[str, Any] = {}
    capsule = object()

    def native_dlpack(_self: pd.DataFrame, **kwds: Any) -> object:
        received.update(kwds)
        return capsule

    monkeypatch.setattr(pd.DataFrame, "__dlpack__", native_dlpack, raising=False)
    monkeypatch.setattr(
        pd.DataFrame, "__dlpack_device__", lambda _self: (2, 0), raising=False
    )
    df = nw.from_native(pd.DataFrame({"a": [1, 2]}), eager_only=True)

    assert df.__dlpack_device__() == (2, 0)
    assert df.__dlpack__(copy=True) is capsule
    assert received == {"copy": True}
