from __future__ import annotations

from typing import Any, NoReturn

import pytest
from packaging.version import Version as PkgVersion

import narwhals as nw
from tests.utils import PYARROW_VERSION

pytest.importorskip("pyarrow")

import pyarrow as pa

pytestmark = pytest.mark.skipif(
    PYARROW_VERSION < PkgVersion("14"), reason="too old for pycapsule in PyArrow"
)

data = [1, 2, None]


def _poison(*_: Any) -> NoReturn:
    msg = "__arrow_c_array__ was poisoned"
    raise RuntimeError(msg)


def test_arrow_c_array(constructor_eager: Any) -> None:
    s = nw.from_native(constructor_eager({"a": data}), eager_only=True)["a"]
    result = pa.array(s)
    assert result.to_pylist() == data


def test_arrow_c_array_is_called(monkeypatch: pytest.MonkeyPatch) -> None:
    # "poison" the dunder method to make sure it actually got called above
    monkeypatch.setattr("narwhals.series.Series.__arrow_c_array__", _poison)
    s = nw.from_native(pa.chunked_array([data]), series_only=True)
    with pytest.raises(RuntimeError, match="poisoned"):
        pa.array(s)


def test_arrow_c_array_requested_schema(constructor_eager: Any) -> None:
    s = nw.from_native(constructor_eager({"a": data}), eager_only=True)["a"]
    result = pa.array(s, type=pa.int32())
    assert result.type == pa.int32()
    assert result.to_pylist() == data


def test_arrow_c_array_pyarrow_too_old(monkeypatch: pytest.MonkeyPatch) -> None:
    s = nw.from_native(pa.chunked_array([data]), series_only=True)
    monkeypatch.setattr("narwhals._utils.backend_version", lambda _: (13,))
    with pytest.raises(ModuleNotFoundError, match=r"pyarrow>=14\.0\.0.*ChunkedArray"):
        s.__arrow_c_array__()
