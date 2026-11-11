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
# numpy<1.24 raises `TypeError` instead of `BufferError` for dtypes DLPack can't hold.
UNSUPPORTED_DTYPE_ERROR = (
    BufferError if NUMPY_VERSION >= PkgVersion("1.24") else TypeError
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
        ([1, 2, 3], np.array([1, 2, 3], dtype="int64")),
        ([1.5, 2.0, 3.0], np.array([1.5, 2.0, 3.0])),
        ([True, False, True], np.array([True, False, True])),
        ([1, None, 3], np.array([1.0, np.nan, 3.0])),
    ],
)
def test_dlpack(
    request: pytest.FixtureRequest,
    constructor_eager: ConstructorEager,
    data: list[Any],
    expected: np.ndarray[Any, Any],
) -> None:
    if expected.dtype == bool and NUMPY_VERSION < PkgVersion("1.25"):  # pragma: no cover
        # numpy<1.25 can't export booleans via DLPack.
        request.applymarker(pytest.mark.xfail(raises=UNSUPPORTED_DTYPE_ERROR))
    s = nw.from_native(constructor_eager({"a": data}), eager_only=True)["a"]
    assert s.__dlpack_device__() == (1, 0)
    np.testing.assert_array_equal(np.from_dlpack(s), expected)


def test_dlpack_legacy_consumer(constructor_eager: ConstructorEager) -> None:
    s = nw.from_native(constructor_eager({"a": [1, 2, 3]}), eager_only=True)["a"]
    result = np.from_dlpack(LegacyConsumerView(s))
    np.testing.assert_array_equal(result, np.array([1, 2, 3]))


def test_dlpack_unsupported_dtype(constructor_eager: ConstructorEager) -> None:
    s = nw.from_native(constructor_eager({"a": ["x", "y"]}), eager_only=True)["a"]
    with pytest.raises(UNSUPPORTED_DTYPE_ERROR):
        np.from_dlpack(s)


def test_dlpack_defers_to_native(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    received: dict[str, Any] = {}
    capsule = object()

    def native_dlpack(_self: pd.Series, **kwds: Any) -> object:
        received.update(kwds)
        return capsule

    monkeypatch.setattr(pd.Series, "__dlpack__", native_dlpack, raising=False)
    monkeypatch.setattr(
        pd.Series, "__dlpack_device__", lambda _self: (2, 0), raising=False
    )
    s = nw.from_native(pd.Series([1, 2, 3]), series_only=True)

    assert s.__dlpack_device__() == (2, 0)
    assert s.__dlpack__(max_version=(1, 0), copy=False) is capsule
    # Unset keywords are not forwarded, so older producers don't reject them.
    assert received == {"max_version": (1, 0), "copy": False}
