from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pizzlies as pz
import pytest

import narwhals as nw

if TYPE_CHECKING:
    from pathlib import Path


def assert_pizzlies_frame(df: Any, expected: dict[str, list[Any]]) -> None:
    assert nw.get_native_namespace(df) is pz
    native = df.to_native()
    assert type(native) is dict
    assert native == expected


def test_reexports_narwhals_api() -> None:
    assert set(nw.__all__) <= set(pz.__all__)
    for name in pz.__all__:
        assert hasattr(pz, name), name


def test_standalone_usage() -> None:
    df = pz.from_dict({"a": [1, 2, 3]})
    result = df.with_columns(b=pz.col("a") * 2)
    assert isinstance(result, pz.DataFrame)
    assert_pizzlies_frame(result, {"a": [1, 2, 3], "b": [2, 4, 6]})


def test_from_dict_schema() -> None:
    df = pz.from_dict({"a": [1, 2]}, schema={"a": pz.Float64()})
    assert df.schema == {"a": pz.Float64()}
    assert_pizzlies_frame(df, {"a": [1.0, 2.0]})


def test_from_dicts() -> None:
    df = pz.from_dicts([{"a": 1, "b": "x"}, {"a": 2, "b": "y"}])
    assert_pizzlies_frame(df, {"a": [1, 2], "b": ["x", "y"]})


def test_from_numpy() -> None:
    np = pytest.importorskip("numpy")
    df = pz.from_numpy(np.array([[1, 2], [3, 4]]), ["a", "b"])
    assert_pizzlies_frame(df, {"a": [1, 3], "b": [2, 4]})


def test_from_arrow() -> None:
    pa = pytest.importorskip("pyarrow")
    df = pz.from_arrow(pa.table({"a": [1, None]}))
    assert_pizzlies_frame(df, {"a": [1, None]})


def test_new_series() -> None:
    s = pz.new_series("a", [1, 2], pz.Float64())
    assert nw.get_native_namespace(s) is pz
    assert s.to_native() == [1.0, 2.0]


def test_read_and_scan_csv(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    path.write_text("a;b\n1;x\n2;y\n")
    expected = {"a": [1, 2], "b": ["x", "y"]}
    assert_pizzlies_frame(pz.read_csv(str(path), separator=";"), expected)
    assert_pizzlies_frame(pz.scan_csv(str(path), separator=";").collect(), expected)


def test_backend_name() -> None:
    df = nw.from_dict({"a": [1]}, backend=pz.BACKEND)
    assert_pizzlies_frame(df, {"a": [1]})
    assert_pizzlies_frame(nw.from_native({"a": [1]}, eager_only=True), {"a": [1]})
