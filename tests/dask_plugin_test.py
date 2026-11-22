from __future__ import annotations

import types
import warnings
from typing import Any, get_args

import pytest

import narwhals as nw
import narwhals.stable.v1 as nw_v1
from narwhals.utils import Version

pytest.importorskip("dask")
pd = pytest.importorskip("pandas")

import dask.dataframe as dd  # noqa: E402

from narwhals._dask.namespace import DaskNamespace  # noqa: E402


def _find_plugin(plugin: Any) -> Any:
    def find(backend_name: str, /) -> Any:
        return plugin if backend_name == "narwhals-dask" else None

    return find


@pytest.fixture
def fake_plugin(monkeypatch: pytest.MonkeyPatch) -> list[Version]:
    """Install a stand-in for `narwhals-dask`, recording the versions it is asked for."""
    calls: list[Version] = []

    def narwhals_namespace(version: Version) -> DaskNamespace:
        calls.append(version)
        return DaskNamespace(version=version)

    plugin = types.ModuleType("fake_narwhals_dask")
    plugin.__narwhals_namespace__ = narwhals_namespace  # type: ignore[attr-defined]
    monkeypatch.setattr("narwhals.plugins._find_plugin", _find_plugin(plugin))
    monkeypatch.setattr("narwhals.plugins._PLUGIN_NAMESPACES", {})
    return calls


def test_dask_warns_without_plugin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("narwhals.plugins._find_plugin", _find_plugin(None))
    native = dd.from_dict({"a": [1, 2]}, npartitions=1)
    with pytest.warns(FutureWarning, match="install it .*narwhals-dask"):
        nw.from_native(native)
    with pytest.warns(FutureWarning, match="install it .*narwhals-dask"):
        nw_v1.from_native(native)
    with pytest.warns(FutureWarning, match="install it .*narwhals-dask"):
        nw.from_native(pd.DataFrame({"a": [1, 2]})).lazy("dask")


def test_dask_uses_plugin(fake_plugin: list[Version]) -> None:
    native = dd.from_dict({"a": [1, 2]}, npartitions=1)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        result = nw.from_native(native).select(nw.col("a").sum()).collect()
        result_v1 = nw_v1.from_native(native).collect()
        nw.from_native(pd.DataFrame({"a": [1, 2]})).lazy("dask")
    assert result.item() == 3
    assert result_v1.to_dict(as_series=False) == {"a": [1, 2]}
    assert fake_plugin == [Version.MAIN, Version.V1]


def test_no_default_is_public() -> None:
    from narwhals._utils import NO_DEFAULT as PRIVATE_NO_DEFAULT
    from narwhals.typing import NoDefault
    from narwhals.utils import NO_DEFAULT

    assert NO_DEFAULT is PRIVATE_NO_DEFAULT
    assert get_args(NoDefault) == (NO_DEFAULT,)
