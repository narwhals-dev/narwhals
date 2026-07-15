from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw
import narwhals.stable.v1.dependencies as nw_v1_dependencies
import narwhals.stable.v2.dependencies as nw_v2_dependencies
from narwhals import dependencies as nw_dependencies
from narwhals._typing import PluginName

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._typing import EagerAllowed, IntoBackend
    from narwhals.plugins import Plugin
    from narwhals.utils import Version

DEPENDENCIES_MODULES = (nw_dependencies, nw_v1_dependencies, nw_v2_dependencies)


class FakeNative:
    """Native object of an imaginary plugin-backed library."""


class FakeCompliantDataFrame:
    def __narwhals_dataframe__(self) -> Self:  # pragma: no cover
        return self


class FakeCompliantLazyFrame:
    def __narwhals_lazyframe__(self) -> Self:  # pragma: no cover
        return self


class FakeCompliantSeries:
    def __narwhals_series__(self) -> Self:  # pragma: no cover
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


def test_plugin_name_runtime() -> None:
    # `PluginName` is a `NewType`: identity at runtime, nominal for type checkers.
    name = PluginName("some-plugin")
    assert name == "some-plugin"
    assert nw.Implementation.from_backend(name) is nw.Implementation.UNKNOWN


if TYPE_CHECKING:
    # Static-only regression guards for `PluginName`
    def typing_backend_plugin_name(
        plugin_name: PluginName,
        dynamic_string: str,
        eager_or_plugin: IntoBackend[EagerAllowed | PluginName],
        df: nw.DataFrame[Any],
    ) -> None:
        data = {"a": [1, 2]}

        # Accepted: an explicitly wrapped plugin name, everything `IntoBackend[EagerAllowed | PluginName]` covers.
        nw.from_dict(data, backend=plugin_name)
        nw.from_dict(data, backend=eager_or_plugin)
        nw.new_series("a", [1, 2], backend=plugin_name)
        nw.scan_csv("file.csv", backend=plugin_name)
        nw.DataFrame.from_dict(data, backend=plugin_name)
        nw.Implementation.from_backend(plugin_name)

        # Rejected: opaque strings do not satisfy `PluginName`.
        nw.from_dict(data, backend=dynamic_string)  # type: ignore[arg-type]
        nw.new_series("a", [1, 2], backend=dynamic_string)  # type: ignore[arg-type]
        nw.Implementation.from_backend(dynamic_string)  # type: ignore[arg-type]

        # Rejected: lazy-only literals on eager constructors (no regression).
        nw.from_dict(data, backend="duckdb")  # type: ignore[arg-type]

        # Rejected: `.lazy` does not dispatch to plugins (yet).
        df.lazy(plugin_name)  # type: ignore[arg-type]
