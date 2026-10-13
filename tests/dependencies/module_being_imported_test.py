from __future__ import annotations

import sys
from importlib.machinery import ModuleSpec
from types import ModuleType
from typing import TYPE_CHECKING, Any

import pytest

from narwhals.dependencies import (
    get_pandas,
    get_polars,
    get_pyarrow,
    is_into_dataframe,
    is_into_series,
    is_pandas_dataframe,
)

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.mark.parametrize(
    ("name", "getter"),
    [("pandas", get_pandas), ("polars", get_polars), ("pyarrow", get_pyarrow)],
)
def test_module_being_imported(
    monkeypatch: pytest.MonkeyPatch, name: str, getter: Callable[[], Any]
) -> None:
    # https://github.com/narwhals-dev/narwhals/issues/4067
    module = ModuleType(name)
    module.__spec__ = spec = ModuleSpec(name, loader=None)
    monkeypatch.setitem(sys.modules, name, module)
    # `importlib` sets this while the module body is running
    monkeypatch.setattr(spec, "_initializing", True, raising=False)
    assert getter() is None
    obj = [1, 2, 3]
    assert not is_into_dataframe(obj)
    assert not is_into_series(obj)
    assert not is_pandas_dataframe(obj)

    monkeypatch.setattr(spec, "_initializing", False)
    assert getter() is module
