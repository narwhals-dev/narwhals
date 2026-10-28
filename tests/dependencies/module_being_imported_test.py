from __future__ import annotations

import importlib
import sys
from importlib.machinery import ModuleSpec
from types import ModuleType
from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw
from narwhals.dependencies import (
    _get_imported_module,
    get_pyspark_connect,
    get_sqlframe,
    is_into_dataframe,
    is_into_series,
    is_pandas_dataframe,
    is_pandas_index,
    is_pyspark_connect_dataframe,
    is_sqlframe_dataframe,
)

if TYPE_CHECKING:
    from collections.abc import Callable
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


@pytest.mark.parametrize("name", ["pandas", "fireducks.pandas"])
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


@pytest.mark.parametrize(
    ("submodule", "is_native"),
    [
        ("fireducks.pandas", is_pandas_dataframe),
        ("pyspark.sql.connect.dataframe", is_pyspark_connect_dataframe),
        ("sqlframe.base.dataframe", is_sqlframe_dataframe),
    ],
)
def test_detection_does_not_import_submodule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    submodule: str,
    is_native: Callable[[Any], bool],
) -> None:
    # The package is imported but its submodule isn't. Detection must not import it:
    # that can be slow, or fail while pandas is being imported (PySpark's checks).
    top = submodule.partition(".")[0]
    parent, leaf = submodule.rsplit(".", 1)
    (tmp_path / f"{leaf}.py").write_text("raise AssertionError('imported')\n")
    package = ModuleType(parent)
    package.__path__ = [str(tmp_path)]
    monkeypatch.setitem(sys.modules, top, ModuleType(top))
    monkeypatch.setitem(sys.modules, parent, package)
    monkeypatch.delitem(sys.modules, submodule, raising=False)

    assert not is_native(object())


@pytest.mark.parametrize(
    ("get_module", "name"),
    [(get_pyspark_connect, "pyspark.sql.connect"), (get_sqlframe, "sqlframe")],
)
def test_get_module(
    monkeypatch: pytest.MonkeyPatch, get_module: Callable[[], Any], name: str
) -> None:
    module = ModuleType(name)
    monkeypatch.setitem(sys.modules, name, module)
    assert get_module() is module
