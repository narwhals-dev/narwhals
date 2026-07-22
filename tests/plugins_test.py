from __future__ import annotations

import types
from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw
import narwhals.stable.v1.dependencies as nw_v1_dependencies
import narwhals.stable.v2.dependencies as nw_v2_dependencies
from narwhals import dependencies as nw_dependencies
from narwhals.exceptions import PluginError
from narwhals.plugins import PluginName
from tests.utils import PYARROW_VERSION

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from types import ModuleType

    import numpy as np
    import pyarrow as pa
    from typing_extensions import Self

    from narwhals._typing import EagerAllowed, IntoBackend
    from narwhals.plugins import Plugin
    from narwhals.typing import NormalizedPath
    from narwhals.utils import Version

plugin_module = pytest.importorskip("test_plugin")

DEPENDENCIES_MODULES = (nw_dependencies, nw_v1_dependencies, nw_v2_dependencies)

BACKEND: Any = "test-plugin"
DATA: dict[str, Any] = {"a": [1, 1, 2], "b": [4, 5, 6]}
ROWS = [{"a": 1, "b": 4}, {"a": 1, "b": 5}, {"a": 2, "b": 6}]


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


def _np_array(data: Any) -> np.ndarray:
    pytest.importorskip("numpy")
    import numpy as np

    return np.asarray(data)  # type: ignore[no-any-return]


def _arrow_table() -> pa.Table:
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    return pa.table(DATA)


@pytest.fixture
def csv_path(tmp_path: Path) -> str:
    path = tmp_path / "file.csv"
    path.write_text("a,b\n1,4\n1,5\n2,6\n", encoding="utf-8")
    return str(path)


@pytest.fixture
def parquet_path(tmp_path: Path) -> str:
    pq = pytest.importorskip("pyarrow.parquet")
    path = str(tmp_path / "file.parquet")
    pq.write_table(_arrow_table(), path)
    return path


def test_plugin_is_lazy() -> None:
    lf = nw.from_native(DATA)  # type: ignore[call-overload]
    assert isinstance(lf, nw.LazyFrame)
    assert lf.columns == ["a", "b"]


def test_not_implemented() -> None:
    lf = nw.from_native(DATA)  # type: ignore[call-overload]
    with pytest.raises(
        NotImplementedError, match="is not implemented for: 'DictLazyFrame'"
    ):
        lf.select(nw.col("a").ewm_mean())


@pytest.mark.parametrize(
    "backend",
    ["test-plugin", "test_plugin", plugin_module],
    ids=["entry-point-name", "module-name", "module"],
)
@pytest.mark.parametrize(
    ("scan_function", "path_fixture"),
    [(nw.scan_csv, "csv_path"), (nw.scan_parquet, "parquet_path")],
    ids=["scan_csv", "scan_parquet"],
)
def test_scan_plugin(
    request: pytest.FixtureRequest,
    scan_function: Callable[..., nw.LazyFrame[Any]],
    path_fixture: str,
    backend: str | ModuleType,
) -> None:
    """`backend` resolves via the entry point name, its module name, or the module itself."""
    lf = scan_function(request.getfixturevalue(path_fixture), backend=backend)
    assert isinstance(lf, nw.LazyFrame)
    assert lf.columns == ["a", "b"]


@pytest.mark.parametrize(
    ("hook", "read_function", "path_fixture"),
    [
        ("read_csv", nw.read_csv, "csv_path"),
        ("read_parquet", nw.read_parquet, "parquet_path"),
    ],
    ids=["read_csv", "read_parquet"],
)
def test_read_plugin_scan_only(
    request: pytest.FixtureRequest,
    hook: str,
    read_function: Callable[..., nw.DataFrame[Any]],
    path_fixture: str,
) -> None:
    """`test_plugin` wraps dicts lazily and only implements `scan_*`, so eager reads raise."""
    with pytest.raises(PluginError, match=f"expected to implement `{hook}`"):
        read_function(request.getfixturevalue(path_fixture), backend=BACKEND)


def _eager_io_plugin() -> types.ModuleType:
    """A plugin whose compliant namespace also implements the `read_*` methods."""
    from test_plugin.dataframe import DictDataFrame
    from test_plugin.namespace import DictNamespace

    class EagerIODictNamespace(DictNamespace):
        def read_csv(
            self, source: NormalizedPath, *, separator: str = ",", **kwds: Any
        ) -> DictDataFrame:
            data = self.scan_csv(source, separator=separator, **kwds)._native_frame
            return DictDataFrame(data, version=self._version)

        def read_parquet(self, source: NormalizedPath, **kwds: Any) -> DictDataFrame:
            data = self.scan_parquet(source, **kwds)._native_frame
            return DictDataFrame(data, version=self._version)

    plugin = types.ModuleType("eager_io_plugin")
    plugin.__narwhals_namespace__ = lambda version: EagerIODictNamespace(  # type: ignore[attr-defined]
        version=version
    )
    return plugin


def test_read_plugin_eager_namespace(csv_path: str, parquet_path: str) -> None:
    """A plugin namespace implementing `read_*` serves eager reads, per the IO contract."""
    plugin = _eager_io_plugin()
    df_csv = nw.read_csv(csv_path, backend=plugin)
    assert isinstance(df_csv, nw.DataFrame)
    assert df_csv.to_native() == {"a": ["1", "1", "2"], "b": ["4", "5", "6"]}
    df_parquet = nw.read_parquet(parquet_path, backend=plugin)
    assert isinstance(df_parquet, nw.DataFrame)
    assert df_parquet.to_native() == DATA


@pytest.mark.parametrize(
    "make_dataframe",
    [
        lambda: nw.from_dict(DATA, backend=BACKEND),
        lambda: nw.from_dicts(ROWS, backend=BACKEND),
        lambda: nw.from_numpy(
            _np_array([[1, 4], [1, 5], [2, 6]]), schema=["a", "b"], backend=BACKEND
        ),
        pytest.param(
            lambda: nw.from_arrow(_arrow_table(), backend=BACKEND),
            marks=pytest.mark.skipif(PYARROW_VERSION < (14,), reason="too old"),
        ),
        lambda: nw.DataFrame.from_dict(DATA, backend=BACKEND),
        lambda: nw.DataFrame.from_dicts(ROWS, backend=BACKEND),
        lambda: nw.DataFrame.from_numpy(
            _np_array([[1, 4], [1, 5], [2, 6]]), schema=["a", "b"], backend=BACKEND
        ),
        pytest.param(
            lambda: nw.DataFrame.from_arrow(_arrow_table(), backend=BACKEND),
            marks=pytest.mark.skipif(PYARROW_VERSION < (14,), reason="too old"),
        ),
    ],
)
def test_eager_dataframe_constructors_plugin(
    make_dataframe: Callable[[], nw.DataFrame[Any]],
) -> None:
    """Eager constructors dispatch to the plugin's `EagerNamespace`-compliant namespace."""
    df = make_dataframe()
    assert isinstance(df, nw.DataFrame)
    assert df.to_native() == DATA


@pytest.mark.parametrize(
    "make_series",
    [
        lambda: nw.new_series("a", [1, 2, 3], backend=BACKEND),
        lambda: nw.Series.from_iterable("a", [1, 2, 3], backend=BACKEND),
        lambda: nw.Series.from_numpy("a", _np_array([1, 2, 3]), backend=BACKEND),
    ],
)
def test_eager_series_constructors_plugin(
    make_series: Callable[[], nw.Series[Any]],
) -> None:
    """Eager constructors dispatch to the plugin's `EagerNamespace`-compliant namespace."""
    s = make_series()
    assert isinstance(s, nw.Series)
    assert s.name == "a"
    assert s.to_native() == [1, 2, 3]


def test_series_scatter_plugin() -> None:
    """`scatter` constructs indices/values via the plugin's own namespace."""
    s = nw.Series.from_iterable("a", [1, 2, 3], backend=BACKEND)
    assert s.scatter([0, 2], [99, 77]).to_native() == [99, 2, 77]
    assert s.scatter(1, 50).to_native() == [1, 50, 3]
    # Original Series is unchanged, and empty indices are a no-op.
    assert s.to_native() == [1, 2, 3]
    assert s.scatter([], []).to_native() == [1, 2, 3]


def test_dataframe_filter_mask_plugin() -> None:
    """`filter(list[bool])` builds the mask series via the plugin's own namespace."""
    df = nw.from_dict(DATA, backend=BACKEND)
    with pytest.raises(NotImplementedError, match="'all_horizontal' is not implemented"):
        df.filter([True, False, True])


@pytest.mark.parametrize(
    "call",
    [
        lambda backend: nw.scan_csv("x.csv", backend=backend),
        lambda backend: nw.read_csv("x.csv", backend=backend),
        lambda backend: nw.scan_parquet("x.parquet", backend=backend),
        lambda backend: nw.read_parquet("x.parquet", backend=backend),
        lambda backend: nw.from_dict(DATA, backend=backend),
    ],
    ids=["scan_csv", "read_csv", "scan_parquet", "read_parquet", "from_dict"],
)
def test_plugin_missing_narwhals_namespace(
    call: Callable[[types.ModuleType], Any],
) -> None:
    """IO and eager functions require the plugin to implement `__narwhals_namespace__`."""
    empty_namespace = types.ModuleType("empty_plugin")
    with pytest.raises(
        PluginError, match="expected to implement `__narwhals_namespace__`"
    ):
        call(empty_namespace)


def _not_implemented_io_namespace() -> Any:
    from narwhals._utils import not_implemented

    class NotImplementedIONamespace:
        scan_csv = not_implemented()
        read_csv = not_implemented()
        scan_parquet = not_implemented()
        read_parquet = not_implemented()

    return NotImplementedIONamespace()


@pytest.mark.parametrize(
    "make_namespace", [object, _not_implemented_io_namespace], ids=["absent", "stubbed"]
)
@pytest.mark.parametrize(
    ("hook", "call"),
    [
        ("scan_csv", lambda backend: nw.scan_csv("x.csv", backend=backend)),
        ("read_csv", lambda backend: nw.read_csv("x.csv", backend=backend)),
        ("scan_parquet", lambda backend: nw.scan_parquet("x.parquet", backend=backend)),
        ("read_parquet", lambda backend: nw.read_parquet("x.parquet", backend=backend)),
    ],
)
def test_plugin_missing_io_method(
    hook: str, call: Callable[[types.ModuleType], Any], make_namespace: Callable[[], Any]
) -> None:
    """A plugin whose compliant namespace lacks the IO method raises an informative PluginError.

    Both a plainly absent method and a `not_implemented` placeholder count as missing.
    """
    minimal_plugin = types.ModuleType("minimal_plugin")
    minimal_plugin.__narwhals_namespace__ = lambda version: make_namespace()  # type: ignore[attr-defined]  # noqa: ARG005
    with pytest.raises(PluginError, match=f"expected to implement `{hook}`"):
        call(minimal_plugin)


def _not_implemented_namespace() -> Any:
    from narwhals._utils import not_implemented

    class LazyOnlyNamespace:
        _series = not_implemented()
        _dataframe = not_implemented()

    return LazyOnlyNamespace()


@pytest.mark.parametrize("make_namespace", [object, _not_implemented_namespace])
@pytest.mark.parametrize(
    "call",
    [
        lambda backend: nw.from_dict(DATA, backend=backend),
        lambda backend: nw.from_dicts(ROWS, backend=backend),
        lambda backend: nw.new_series("a", [1], backend=backend),
        lambda backend: nw.Series.from_iterable("a", [1], backend=backend),
        lambda backend: nw.DataFrame.from_dict(DATA, backend=backend),
    ],
)
def test_plugin_not_eager_allowed(
    call: Callable[[types.ModuleType], Any], make_namespace: Callable[[], Any]
) -> None:
    """Eager functions require an `EagerNamespace`-compliant plugin namespace."""
    lazy_plugin = types.ModuleType("lazy_plugin")
    lazy_plugin.__narwhals_namespace__ = lambda version: make_namespace()  # type: ignore[attr-defined]  # noqa: ARG005
    with pytest.raises(PluginError, match="does not provide eager support"):
        call(lazy_plugin)


def test_unknown_backend_raises() -> None:
    """A string matching neither a built-in backend nor an installed plugin."""
    with pytest.raises(ValueError, match="Unsupported backend: 'not-a-backend'"):
        nw.scan_csv("x.csv", backend="not-a-backend")  # type: ignore[arg-type]


def test_from_native_unsupported_object() -> None:
    """An object no installed plugin recognises falls through to the unsupported-type error."""
    with pytest.raises(TypeError, match="Unsupported dataframe type"):
        nw.from_native(object())  # type: ignore[call-overload]


def test_is_into_lazyframe() -> None:
    # https://github.com/narwhals-dev/narwhals/issues/3714
    df_native = {"a": [1, 1, 2], "b": [4, 5, 6]}
    for dependencies in DEPENDENCIES_MODULES:
        assert dependencies.is_into_lazyframe(df_native)


def test_is_into_dataframe() -> None:
    # `test_plugin` converts to a LazyFrame, so `is_into_dataframe` should not match.
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
        lf = df.lazy()
        # Rejected: `.collect` does not dispatch to plugins (yet).
        lf.collect(plugin_name)  # type: ignore[arg-type]
