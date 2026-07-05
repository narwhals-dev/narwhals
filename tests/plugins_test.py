from __future__ import annotations

import types
from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from narwhals.plugins import Plugin

DATA = {"a": [1, 1, 2], "b": [4, 5, 6]}


@pytest.fixture
def csv_path(tmp_path: Path) -> str:
    path = tmp_path / "file.csv"
    path.write_text("a,b\n1,4\n1,5\n2,6\n", encoding="utf-8")
    return str(path)


@pytest.fixture
def parquet_path(tmp_path: Path) -> str:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    path = str(tmp_path / "file.parquet")
    pq.write_table(pa.table(DATA), path)
    return path


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


def test_typing() -> None:
    pytest.importorskip("test_plugin")
    import test_plugin

    _plugin: Plugin = test_plugin


@pytest.mark.parametrize("backend", ["test-plugin", "test_plugin"])
def test_scan_csv_plugin_backend_name(csv_path: str, backend: str) -> None:
    """Both the entry point name and its module resolve to the plugin."""
    pytest.importorskip("test_plugin")
    lf = nw.scan_csv(csv_path, backend=backend)  # type: ignore[arg-type]
    assert isinstance(lf, nw.LazyFrame)
    assert lf.columns == ["a", "b"]


def test_scan_csv_plugin_backend_module(csv_path: str) -> None:
    test_plugin = pytest.importorskip("test_plugin")
    lf = nw.scan_csv(csv_path, backend=test_plugin)
    assert isinstance(lf, nw.LazyFrame)
    assert lf.columns == ["a", "b"]


def test_scan_parquet_plugin(parquet_path: str) -> None:
    pytest.importorskip("test_plugin")
    lf = nw.scan_parquet(parquet_path, backend="test-plugin")  # type: ignore[arg-type]
    assert isinstance(lf, nw.LazyFrame)
    assert lf.columns == ["a", "b"]


def test_read_plugin_lazy_only(csv_path: str, parquet_path: str) -> None:
    """`test_plugin.from_native` wraps dicts lazily, so eager reads raise."""
    pytest.importorskip("test_plugin")
    with pytest.raises(TypeError, match="Cannot only use `eager_only`"):
        nw.read_csv(csv_path, backend="test-plugin")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Cannot only use `eager_only`"):
        nw.read_parquet(parquet_path, backend="test-plugin")  # type: ignore[arg-type]


def test_from_dict_plugin() -> None:
    pytest.importorskip("test_plugin")
    df = nw.from_dict(DATA, backend="test-plugin")  # type: ignore[arg-type]
    assert isinstance(df, nw.DataFrame)
    assert df.to_native() == DATA


def test_from_dicts_plugin() -> None:
    pytest.importorskip("test_plugin")
    rows = [{"a": 1, "b": 4}, {"a": 2, "b": 5}]
    df = nw.from_dicts(rows, backend="test-plugin")  # type: ignore[arg-type]
    assert isinstance(df, nw.DataFrame)
    assert df.to_native() == {"a": [1, 2], "b": [4, 5]}


def test_from_numpy_plugin() -> None:
    pytest.importorskip("test_plugin")
    np = pytest.importorskip("numpy")
    arr = np.array([[1, 4], [2, 5]])
    df = nw.from_numpy(arr, schema=["a", "b"], backend="test-plugin")  # type: ignore[arg-type]
    assert isinstance(df, nw.DataFrame)
    assert df.to_native() == {"a": [1, 2], "b": [4, 5]}


def test_from_arrow_plugin() -> None:
    pytest.importorskip("test_plugin")
    pa = pytest.importorskip("pyarrow")
    df = nw.from_arrow(pa.table(DATA), backend="test-plugin")  # type: ignore[arg-type]
    assert isinstance(df, nw.DataFrame)
    assert df.to_native() == DATA


def test_new_series_plugin() -> None:
    pytest.importorskip("test_plugin")
    s = nw.new_series("a", [1, 2, 3], backend="test-plugin")  # type: ignore[arg-type]
    assert isinstance(s, nw.Series)
    assert s.name == "a"
    assert s.to_native() == [1, 2, 3]


def test_dataframe_classmethods_plugin() -> None:
    pytest.importorskip("test_plugin")
    np = pytest.importorskip("numpy")
    pa = pytest.importorskip("pyarrow")
    expected = {"a": [1, 2], "b": [4, 5]}

    df = nw.DataFrame.from_dict(expected, backend="test-plugin")  # type: ignore[arg-type]
    assert df.to_native() == expected
    df = nw.DataFrame.from_dicts(
        [{"a": 1, "b": 4}, {"a": 2, "b": 5}],
        backend="test-plugin",  # type: ignore[arg-type]
    )
    assert df.to_native() == expected
    df = nw.DataFrame.from_numpy(
        np.array([[1, 4], [2, 5]]),
        schema=["a", "b"],
        backend="test-plugin",  # type: ignore[arg-type]
    )
    assert df.to_native() == expected
    df = nw.DataFrame.from_arrow(pa.table(expected), backend="test-plugin")  # type: ignore[arg-type]
    assert df.to_native() == expected


def test_series_classmethods_plugin() -> None:
    pytest.importorskip("test_plugin")
    np = pytest.importorskip("numpy")

    s = nw.Series.from_iterable("a", [1, 2, 3], backend="test-plugin")  # type: ignore[arg-type]
    assert isinstance(s, nw.Series)
    assert s.name == "a"
    assert s.to_native() == [1, 2, 3]

    s = nw.Series.from_numpy("a", np.array([1, 2, 3]), backend="test-plugin")  # type: ignore[arg-type]
    assert s.name == "a"
    assert s.to_native() == [1, 2, 3]


def test_series_scatter_plugin() -> None:
    """`scatter` constructs indices/values via the plugin's own namespace."""
    pytest.importorskip("test_plugin")
    s = nw.Series.from_iterable("a", [1, 2, 3], backend="test-plugin")  # type: ignore[arg-type]
    assert s.scatter([0, 2], [99, 77]).to_native() == [99, 2, 77]
    assert s.scatter(1, 50).to_native() == [1, 50, 3]
    # Original Series is unchanged, and empty indices are a no-op.
    assert s.to_native() == [1, 2, 3]
    assert s.scatter([], []).to_native() == [1, 2, 3]


def test_dataframe_filter_mask_plugin() -> None:
    """`filter(list[bool])` builds the mask series via the plugin's own namespace.

    The minimal test-plugin implements no expression layer, so the call proceeds past
    mask construction and fails in narwhals' `all_horizontal` instead of erroring on
    plugin resolution.
    """
    pytest.importorskip("test_plugin")
    df = nw.from_dict(DATA, backend="test-plugin")  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError, match="'all_horizontal' is not implemented"):
        df.filter([True, False, True])


def test_plugin_not_eager_capable_not_implemented_namespace() -> None:
    """`_series`/`_dataframe` may be `not_implemented` descriptors, which raise on access."""
    from narwhals._utils import not_implemented

    class LazyOnlyNamespace:
        _series = not_implemented()
        _dataframe = not_implemented()

    lazy_plugin = types.ModuleType("lazy_plugin")
    lazy_plugin.__narwhals_namespace__ = lambda version: LazyOnlyNamespace()  # type: ignore[attr-defined]  # noqa: ARG005
    with pytest.raises(ValueError, match="does not provide eager support"):
        nw.from_dict(DATA, backend=lazy_plugin)


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
    """A module without the required IO hook raises an informative AttributeError."""
    empty_namespace = types.ModuleType("empty_plugin")
    with pytest.raises(AttributeError, match=f"expected to implement `{hook}` function"):
        call(empty_namespace)


def test_plugin_missing_narwhals_namespace() -> None:
    """Eager functions require the plugin to implement `__narwhals_namespace__`."""
    empty_namespace = types.ModuleType("empty_plugin")
    with pytest.raises(
        AttributeError, match="expected to implement `__narwhals_namespace__`"
    ):
        nw.from_dict(DATA, backend=empty_namespace)


def test_plugin_not_eager_capable() -> None:
    """Eager functions require an `EagerNamespace`-compliant plugin namespace."""
    lazy_namespace = types.ModuleType("lazy_plugin")
    lazy_namespace.__narwhals_namespace__ = lambda version: object()  # type: ignore[attr-defined]  # noqa: ARG005
    for call in (
        lambda: nw.from_dict(DATA, backend=lazy_namespace),
        lambda: nw.from_dicts([{"a": 1}], backend=lazy_namespace),
        lambda: nw.new_series("a", [1], backend=lazy_namespace),
        lambda: nw.Series.from_iterable("a", [1], backend=lazy_namespace),
        lambda: nw.DataFrame.from_dict(DATA, backend=lazy_namespace),
    ):
        with pytest.raises(ValueError, match="does not provide eager support"):
            call()


def test_unknown_backend_raises() -> None:
    """A string matching neither a built-in backend nor an installed plugin."""
    with pytest.raises(ValueError, match="Unsupported backend: 'not-a-backend'"):
        nw.scan_csv("x.csv", backend="not-a-backend")  # type: ignore[arg-type]
