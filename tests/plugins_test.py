from __future__ import annotations

import types
from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw
import narwhals.stable.v1.dependencies as nw_v1_dependencies
import narwhals.stable.v2.dependencies as nw_v2_dependencies
from narwhals import dependencies as nw_dependencies
from narwhals.exceptions import PluginError
from tests.utils import PYARROW_VERSION

plugin_module = pytest.importorskip("test_plugin")

from test_plugin.dataframe import DictDataFrame, DictLazyFrame  # noqa: E402
from test_plugin.series import DictSeries  # noqa: E402

DEPENDENCIES_MODULES = (nw_dependencies, nw_v1_dependencies, nw_v2_dependencies)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from types import ModuleType

    import numpy as np
    import pyarrow as pa

    from narwhals.plugins import Plugin

BACKEND: Any = "test-plugin"
DATA: dict[str, Any] = {"a": [1, 1, 2], "b": [4, 5, 6]}
ROWS = [{"a": 1, "b": 4}, {"a": 1, "b": 5}, {"a": 2, "b": 6}]


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


def test_typing() -> None:
    import test_plugin

    _plugin: Plugin = test_plugin


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
    ("read_function", "path_fixture"),
    [(nw.read_csv, "csv_path"), (nw.read_parquet, "parquet_path")],
    ids=["read_csv", "read_parquet"],
)
def test_read_plugin_lazy_only(
    request: pytest.FixtureRequest,
    read_function: Callable[..., nw.DataFrame[Any]],
    path_fixture: str,
) -> None:
    """`test_plugin.from_native` wraps dicts lazily, so eager reads raise."""
    with pytest.raises(TypeError, match="Cannot only use `eager_only`"):
        read_function(request.getfixturevalue(path_fixture), backend=BACKEND)


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
    ("hook", "call"),
    [
        ("scan_csv", lambda backend: nw.scan_csv("x.csv", backend=backend)),
        ("read_csv", lambda backend: nw.read_csv("x.csv", backend=backend)),
        ("scan_parquet", lambda backend: nw.scan_parquet("x.parquet", backend=backend)),
        ("read_parquet", lambda backend: nw.read_parquet("x.parquet", backend=backend)),
    ],
)
def test_plugin_missing_io_hook(
    hook: str, call: Callable[[types.ModuleType], Any]
) -> None:
    """A module without the required IO hook raises an informative PluginError."""
    empty_namespace = types.ModuleType("empty_plugin")
    with pytest.raises(PluginError, match=f"expected to implement `{hook}` function"):
        call(empty_namespace)


def test_plugin_missing_narwhals_namespace() -> None:
    """Eager functions require the plugin to implement `__narwhals_namespace__`."""
    empty_namespace = types.ModuleType("empty_plugin")
    with pytest.raises(
        PluginError, match="expected to implement `__narwhals_namespace__`"
    ):
        nw.from_dict(DATA, backend=empty_namespace)


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
    df_native = {"a": [1, 1, 2], "b": [4, 5, 6]}
    for dependencies in DEPENDENCIES_MODULES:
        assert dependencies.is_into_lazyframe(df_native)


def test_is_into_dataframe() -> None:
    df_native = {"a": [1, 1, 2], "b": [4, 5, 6]}
    for dependencies in DEPENDENCIES_MODULES:
        assert not dependencies.is_into_dataframe(df_native)


@pytest.mark.parametrize(
    ("compliant_cls", "expected_kind"),
    [(DictDataFrame, "dataframe"), (DictLazyFrame, "lazyframe"), (DictSeries, "series")],
)
def test_is_into_mocked_plugin(compliant_cls: type, expected_kind: str) -> None:
    for dep in DEPENDENCIES_MODULES:
        assert dep.is_into_dataframe(compliant_cls) is (expected_kind == "dataframe")
        assert dep.is_into_lazyframe(compliant_cls) is (expected_kind == "lazyframe")
        assert dep.is_into_series(compliant_cls) is (expected_kind == "series")
