from __future__ import annotations

import importlib
import sys
from importlib.machinery import ModuleSpec
from types import ModuleType
from typing import TYPE_CHECKING

import pytest

import narwhals as nw
from narwhals.dependencies import (
    _get_imported_module,
    is_into_dataframe,
    is_into_series,
    is_pandas_index,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_get_imported_module_during_real_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Fails if CPython stops setting the private `__spec__._initializing` we rely on.
    name = "narwhals_module_being_imported_probe"
    (tmp_path / f"{name}.py").write_text(
        "from narwhals.dependencies import _get_imported_module\n"
        "SEEN_DURING_IMPORT = _get_imported_module(__name__)\n"
    )
    monkeypatch.syspath_prepend(tmp_path)
    try:
        module = importlib.import_module(name)
        assert module.SEEN_DURING_IMPORT is None
        assert _get_imported_module(name) is module
    finally:
        sys.modules.pop(name, None)


@pytest.mark.parametrize("name", ["pandas", "fireducks"])
def test_detection_skips_module_being_imported(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    # https://github.com/narwhals-dev/narwhals/issues/4067
    module = ModuleType(name)
    module.__spec__ = ModuleSpec(name, loader=None)
    module.__spec__._initializing = True  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, name, module)

    obj = object()
    assert not is_into_dataframe(obj)
    assert not is_into_series(obj)
    assert not is_pandas_index(obj)
    assert nw.from_native(obj, pass_through=True) is obj
