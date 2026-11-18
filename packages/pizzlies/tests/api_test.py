from __future__ import annotations

from io import BytesIO
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


def test_from_dicts_schema() -> None:
    df = pz.from_dicts([{"a": 1}, {}], schema={"a": pz.Float64()})
    assert df.schema == {"a": pz.Float64()}
    assert_pizzlies_frame(df, {"a": [1.0, None]})


def test_from_numpy() -> None:
    np = pytest.importorskip("numpy")
    df = pz.from_numpy(np.array([[1, 2], [3, 4]]), ["a", "b"])
    assert_pizzlies_frame(df, {"a": [1, 3], "b": [2, 4]})


def test_from_arrow() -> None:
    pa = pytest.importorskip("pyarrow")
    df = pz.from_arrow(pa.table({"a": [1, None]}))
    assert_pizzlies_frame(df, {"a": [1, None]})
    with pytest.raises(nw.exceptions.DuplicateError):
        pz.from_arrow(pa.table([[1], [2]], names=["a", "a"]))


def test_new_series() -> None:
    s = pz.new_series("a", [1, 2], pz.Float64())
    assert nw.get_native_namespace(s) is pz
    assert s.dtype == pz.Float64()
    assert s.to_native() == [1.0, 2.0]


def test_read_and_scan_csv(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    path.write_text("a;b\n1;x\n2;y\n")
    expected = {"a": [1, 2], "b": ["x", "y"]}
    assert_pizzlies_frame(pz.read_csv(str(path), separator=";"), expected)
    assert_pizzlies_frame(pz.scan_csv(str(path), separator=";").collect(), expected)


def test_write_csv_round_trips_line_breaks() -> None:
    data = {"a": ["one\rtwo", "three\nfour", "five\r\nsix"], "b": [1, 2, 3]}
    csv = pz.from_dict(data).write_csv()
    assert csv == 'a,b\n"one\rtwo",1\n"three\nfour",2\n"five\r\nsix",3\n'
    assert_pizzlies_frame(pz.read_csv(BytesIO(csv.encode())), data)


def test_pivot_output_names() -> None:
    df = pz.from_dict(
        {"x": [1, 1, 1], "y": [2020, None, 2021], "z": ["a", "b", "c"], "v": [1, 2, 3]}
    )
    assert df.pivot("y", index="x", values="v").columns == ["x", "2020", "null", "2021"]
    assert df.pivot(["y", "z"], index="x", values="v").columns == [
        "x",
        '{2020,"a"}',
        '{null,"b"}',
        '{2021,"c"}',
    ]
    df = pz.from_dict({"id": ["x"], "category": ["id"], "value": [7]})
    with pytest.raises(nw.exceptions.DuplicateError):
        df.pivot("category", index="id", values="value")


def test_corr_tiny_values() -> None:
    df = pz.from_dict({"a": [1e-100, 2e-100, 3e-100], "b": [1e-100, 2e-100, 3e-100]})
    assert df.select(pz.corr("a", "b")).item() == pytest.approx(1.0)


def test_lazy_from_native_round_trip() -> None:
    lf = pz.from_dict({"a": [1, 2]}).lazy().filter(pz.col("a") > 1)
    assert_pizzlies_frame(nw.from_native(lf.to_native()).collect(), {"a": [2]})


def test_backend_name() -> None:
    df = nw.from_dict({"a": [1]}, backend=pz.BACKEND)
    assert_pizzlies_frame(df, {"a": [1]})
    assert_pizzlies_frame(nw.from_native({"a": [1]}, eager_only=True), {"a": [1]})
