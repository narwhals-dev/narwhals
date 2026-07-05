from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw
import narwhals.stable.v1.dependencies as nw_v1_dependencies
import narwhals.stable.v2.dependencies as nw_v2_dependencies
from narwhals import dependencies as nw_dependencies

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals.plugins import Plugin
    from narwhals.utils import Version

DEPENDENCIES_MODULES = (nw_dependencies, nw_v1_dependencies, nw_v2_dependencies)


class FakeNative:
    """Native object of an imaginary plugin-backed library."""


class FakeCompliantDataFrame:
    def __narwhals_dataframe__(self) -> Self:
        return self


class FakeCompliantLazyFrame:
    def __narwhals_lazyframe__(self) -> Self:
        return self


class FakeCompliantSeries:
    def __narwhals_series__(self) -> Self:
        return self


class FakeNamespace:
    def __init__(self, compliant_cls: type, version: Version) -> None:
        self._compliant_cls = compliant_cls
        self._version = version

    def from_native(self, native_object: object) -> Any:
        assert isinstance(native_object, FakeNative)
        return self._compliant_cls()


class FakePlugin:
    NATIVE_PACKAGE = "builtins"

    def __init__(self, compliant_cls: type) -> None:
        self._compliant_cls = compliant_cls

    def is_native(self, native_object: object) -> bool:
        return isinstance(native_object, FakeNative)

    def __narwhals_namespace__(self, version: Version) -> FakeNamespace:
        return FakeNamespace(self._compliant_cls, version)


class FakeEntryPoint:
    def __init__(self, plugin: FakePlugin) -> None:
        self._plugin = plugin

    def load(self) -> FakePlugin:
        return self._plugin


def test_plugin() -> None:
    pytest.importorskip("test_plugin")
    df_native = {"a": [1, 1, 2], "b": [4, 5, 6]}
    lf = nw.from_native(df_native)  # type: ignore[call-overload]
    assert isinstance(lf, nw.LazyFrame)
    assert lf.columns == ["a", "b"]


def test_not_implemented() -> None:
    pytest.importorskip("test_plugin")
    df_native = {"a": [1, 1, 2], "b": [4, 5, 6]}
    lf = nw.from_native(df_native)  # type: ignore[call-overload]
    with pytest.raises(
        NotImplementedError, match="is not implemented for: 'DictLazyFrame'"
    ):
        lf.select(nw.col("a").ewm_mean())


def test_is_into_lazyframe() -> None:
    # https://github.com/narwhals-dev/narwhals/issues/3714
    pytest.importorskip("test_plugin")
    df_native = {"a": [1, 1, 2], "b": [4, 5, 6]}
    for dependencies in DEPENDENCIES_MODULES:
        assert dependencies.is_into_lazyframe(df_native)


def test_is_into_dataframe() -> None:
    # `test_plugin` converts to a LazyFrame, so `is_into_dataframe` should not match.
    pytest.importorskip("test_plugin")
    df_native = {"a": [1, 1, 2], "b": [4, 5, 6]}
    for dependencies in DEPENDENCIES_MODULES:
        assert not dependencies.is_into_dataframe(df_native)


@pytest.mark.parametrize(
    ("compliant_cls", "expected_kind"),
    [
        (FakeCompliantDataFrame, "dataframe"),
        (FakeCompliantLazyFrame, "lazyframe"),
        (FakeCompliantSeries, "series"),
    ],
)
def test_is_into_mocked_plugin(
    monkeypatch: pytest.MonkeyPatch, compliant_cls: type, expected_kind: str
) -> None:
    from narwhals import plugins

    monkeypatch.setattr(
        plugins,
        "_discover_entrypoints",
        lambda: (FakeEntryPoint(FakePlugin(compliant_cls)),),
    )
    native = FakeNative()
    for dependencies in DEPENDENCIES_MODULES:
        assert dependencies.is_into_dataframe(native) is (expected_kind == "dataframe")
        assert dependencies.is_into_lazyframe(native) is (expected_kind == "lazyframe")
        assert dependencies.is_into_series(native) is (expected_kind == "series")


def test_typing() -> None:
    pytest.importorskip("test_plugin")
    import test_plugin

    _plugin: Plugin = test_plugin
