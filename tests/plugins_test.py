from __future__ import annotations

import re
import types
from functools import partial
from typing import TYPE_CHECKING, Any, Protocol, cast, get_args

import pytest

import narwhals as nw
import narwhals.stable.v1.dependencies as nw_v1_dependencies
import narwhals.stable.v2.dependencies as nw_v2_dependencies
from narwhals import dependencies as nw_dependencies
from narwhals._compliant import CompliantNamespace
from narwhals._utils import EAGER_HINT_EXAMPLES, EagerFunctionName, not_implemented
from narwhals.exceptions import PluginError
from narwhals.plugins import PluginName
from tests.utils import PYARROW_VERSION

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path
    from types import ModuleType
    from typing import TypeAlias

    import pyarrow as pa
    from typing_extensions import Self

    from narwhals._typing import Backend, EagerAllowed, IntoBackend
    from narwhals.plugins import Plugin
    from narwhals.typing import _1DArray, _2DArray
    from narwhals.utils import Version

    _ConstructorData: TypeAlias = "Mapping[str, Any] | Sequence[Mapping[str, Any]] | Callable[[], _2DArray | pa.Table]"

plugin_module = pytest.importorskip("test_plugin")

DEPENDENCIES_MODULES = (nw_dependencies, nw_v1_dependencies, nw_v2_dependencies)

BACKEND = PluginName("test-plugin")
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


class PluginModule(types.ModuleType):
    """An ad-hoc, *deliberately incomplete* plugin module.

    Real plugins live in `packages/`, but the error paths below need namespaces which
    violate the contract on purpose, so they cannot be packaged.
    """

    _factory: Callable[[], object]

    def __init__(self, name: str, factory: Callable[[], object]) -> None:
        super().__init__(name)
        self._factory = factory

    def __narwhals_namespace__(self, version: Version) -> object:
        return self._factory()


class BackendFn(Protocol):
    """A narwhals function whose remaining argument is `backend`."""

    def __call__(
        self, *, backend: IntoBackend[Backend | PluginName]
    ) -> nw.DataFrame[Any] | nw.LazyFrame[Any]: ...


class NotImplementedNamespace(CompliantNamespace[Any, Any]):
    """A namespace which declares, but does not provide, the optional plugin methods.

    `not_implemented` descriptors exist statically yet raise on instance access, so they
    must not be mistaken for support.
    """

    scan_csv = not_implemented()
    read_csv = not_implemented()
    scan_parquet = not_implemented()
    read_parquet = not_implemented()
    _series = not_implemented()
    _dataframe = not_implemented()


def _np_2d_array() -> _2DArray:
    pytest.importorskip("numpy")
    import numpy as np

    return cast("_2DArray", np.array([[1, 4], [1, 5], [2, 6]]))


def _np_1d_array() -> _1DArray:
    pytest.importorskip("numpy")
    import numpy as np

    return cast("_1DArray", np.array([1, 2, 3]))


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
    ("read_function", "path_fixture", "expected"),
    [
        (nw.read_csv, "csv_path", {"a": ["1", "1", "2"], "b": ["4", "5", "6"]}),
        (nw.read_parquet, "parquet_path", DATA),
    ],
    ids=["read_csv", "read_parquet"],
)
def test_read_plugin(
    request: pytest.FixtureRequest,
    read_function: Callable[..., nw.DataFrame[Any]],
    path_fixture: str,
    expected: dict[str, Any],
) -> None:
    """`read_*` dispatch to the namespace's eager half of the IO contract."""
    df = read_function(request.getfixturevalue(path_fixture), backend=BACKEND)
    assert isinstance(df, nw.DataFrame)
    assert df.to_native() == expected


@pytest.mark.parametrize(
    ("dataframe_constructor", "data", "kwargs"),
    [
        (nw.from_dict, DATA, {}),
        (nw.from_dicts, ROWS, {}),
        (nw.from_numpy, _np_2d_array, {"schema": ["a", "b"]}),
        pytest.param(
            nw.from_arrow,
            _arrow_table,
            {},
            marks=pytest.mark.skipif(PYARROW_VERSION < (14,), reason="too old"),
        ),
        (nw.DataFrame.from_dict, DATA, {}),
        (nw.DataFrame.from_dicts, ROWS, {}),
        (nw.DataFrame.from_numpy, _np_2d_array, {"schema": ["a", "b"]}),
        pytest.param(
            nw.DataFrame.from_arrow,
            _arrow_table,
            {},
            marks=pytest.mark.skipif(PYARROW_VERSION < (14,), reason="too old"),
        ),
    ],
)
def test_eager_dataframe_constructors_plugin(
    dataframe_constructor: Callable[..., nw.DataFrame[Any]],
    data: _ConstructorData,
    kwargs: dict[str, Any],
) -> None:
    """Eager constructors dispatch to the plugin's `EagerNamespace`-compliant namespace."""
    # Factories are deferred, so that `importorskip` runs at test time.
    native_data = data() if callable(data) else data
    df = dataframe_constructor(native_data, backend=BACKEND, **kwargs)
    assert isinstance(df, nw.DataFrame)
    assert df.to_native() == DATA


@pytest.mark.parametrize(
    ("series_constructor", "values"),
    [
        (nw.new_series, [1, 2, 3]),
        (nw.Series.from_iterable, [1, 2, 3]),
        (nw.Series.from_numpy, _np_1d_array),
    ],
)
def test_eager_series_constructors_plugin(
    series_constructor: Callable[..., nw.Series[Any]],
    values: list[int] | Callable[[], _1DArray],
) -> None:
    """Eager constructors dispatch to the plugin's `EagerNamespace`-compliant namespace."""
    # The factory is deferred, so that `importorskip` runs at test time.
    native_values = values() if callable(values) else values
    s = series_constructor("a", native_values, backend=BACKEND)
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
    "function",
    [
        partial(nw.scan_csv, "x.csv"),
        partial(nw.read_csv, "x.csv"),
        partial(nw.scan_parquet, "x.parquet"),
        partial(nw.read_parquet, "x.parquet"),
        partial(nw.from_dict, DATA),
    ],
)
def test_plugin_missing_narwhals_namespace(function: BackendFn) -> None:
    """IO and eager functions require the plugin to implement `__narwhals_namespace__`."""
    empty_namespace = types.ModuleType("empty_plugin")
    with pytest.raises(
        PluginError, match="expected to implement `__narwhals_namespace__`"
    ):
        function(backend=empty_namespace)


@pytest.mark.parametrize("make_namespace", [object, NotImplementedNamespace])
@pytest.mark.parametrize(
    ("io_function", "source"),
    [
        (nw.scan_csv, "x.csv"),
        (nw.read_csv, "x.csv"),
        (nw.scan_parquet, "x.parquet"),
        (nw.read_parquet, "x.parquet"),
    ],
)
def test_plugin_missing_io_method(
    io_function: Callable[..., nw.DataFrame[Any] | nw.LazyFrame[Any]],
    source: str,
    make_namespace: Callable[[], object],
) -> None:
    """A plugin whose compliant namespace lacks the IO method raises an informative PluginError.

    Both a plainly absent method and a `not_implemented` placeholder count as missing.
    """
    minimal_plugin = PluginModule("minimal_plugin", make_namespace)
    with pytest.raises(
        PluginError, match=f"expected to implement `{io_function.__name__}`"
    ):
        io_function(source, backend=minimal_plugin)


@pytest.mark.parametrize("make_namespace", [object, NotImplementedNamespace])
@pytest.mark.parametrize(
    "function",
    [
        partial(nw.from_dict, DATA),
        partial(nw.from_dicts, ROWS),
        partial(nw.new_series, "a", [1]),
        partial(nw.Series.from_iterable, "a", [1]),
        partial(nw.DataFrame.from_dict, DATA),
    ],
)
def test_plugin_not_eager_allowed(
    function: Callable[..., nw.DataFrame[Any] | nw.Series[Any]],
    make_namespace: Callable[[], object],
) -> None:
    """Eager functions require an `EagerNamespace`-compliant plugin namespace."""
    lazy_plugin = PluginModule("lazy_plugin", make_namespace)
    with pytest.raises(PluginError, match="does not provide eager support"):
        function(backend=lazy_plugin)


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

    def fake_entrypoints() -> tuple[FakeEntryPoint, ...]:
        return (FakeEntryPoint(FakePlugin(compliant_cls)),)

    monkeypatch.setattr(plugins, "_discover_entrypoints", fake_entrypoints)
    native = FakeNative()
    for dependencies in DEPENDENCIES_MODULES:
        assert dependencies.is_into_dataframe(native) is (expected_kind == "dataframe")
        assert dependencies.is_into_lazyframe(native) is (expected_kind == "lazyframe")
        assert dependencies.is_into_series(native) is (expected_kind == "series")


def test_typing() -> None:
    import test_plugin

    _plugin: Plugin = test_plugin


def test_eager_hint_examples_exhaustive() -> None:
    assert set(get_args(EagerFunctionName)) == set(EAGER_HINT_EXAMPLES)


@pytest.mark.parametrize(
    ("function", "function_name"),
    [
        (partial(nw.from_dict, DATA), "from_dict"),
        (partial(nw.new_series, "a", [1]), "new_series"),
        (partial(nw.DataFrame.from_dicts, ROWS), "DataFrame.from_dicts"),
    ],
)
def test_eager_only_lazy_backend_hint(
    function: Callable[..., Any], function_name: str
) -> None:
    """A lazy-only *built-in* backend gets the per-function hint, keyed by `function_name`."""
    hint = EAGER_HINT_EXAMPLES[function_name]  # type: ignore[index]
    with pytest.raises(ValueError, match=re.escape(f"    {hint}.lazy(")):
        function(backend="duckdb")


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
