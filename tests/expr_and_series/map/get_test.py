from __future__ import annotations

import datetime as dt
import re
from typing import TYPE_CHECKING, Any

import pytest
from packaging.version import Version as PkgVersion

import narwhals as nw
from narwhals.exceptions import InvalidOperationError
from tests.utils import PANDAS_VERSION, POLARS_VERSION, assert_equal_data

if TYPE_CHECKING:
    from tests.utils import Constructor, ConstructorEager

string_data = [{"x": 1, "y": 2}, {"y": 3}, {}, None]
int_data = [{10: 1, 20: 2}, {20: 3}, {}, None]
expected = {"x": [1, None, None, None], "y": [2, 3, None, None]}


def map_frame(
    request: pytest.FixtureRequest,
    constructor: Constructor,
    data: list[Any] = string_data,
    value_type: Any = None,
) -> Any:
    """Build a native frame with a `Map(String | Int64, Int64)` column `a`.

    `value_type` overrides the Arrow value type; Spark-like backends ignore it.
    """
    name = str(constructor)
    if any(backend in name for backend in ("dask", "cudf")):
        request.applymarker(pytest.mark.xfail(reason="no native Map dtype"))
        return constructor({"a": data})
    if "polars" in name and POLARS_VERSION < PkgVersion("2"):  # pragma: no cover
        pytest.skip(reason="native Map dtype requires Polars 2.0.0")
    if "pandas" in name and PANDAS_VERSION < PkgVersion("2.2"):  # pragma: no cover
        pytest.skip(reason="pandas too old for pyarrow types")
    if "sqlframe" in name or "pyspark" in name:
        # Spark infers `MapType` from dicts, but can't read Arrow maps.
        return constructor({"a": data})
    pa = pytest.importorskip("pyarrow")
    key_type = pa.int64() if data is int_data else pa.string()
    map_type = pa.map_(key_type, pa.int64() if value_type is None else value_type)
    table = pa.table({"a": pa.array(data, type=map_type)})
    if "pandas" in name or "modin" in name:
        pd = pytest.importorskip("pandas")
        df = table.to_pandas(types_mapper=pd.ArrowDtype)
        if "modin" in name:
            import modin.pandas as mpd

            return mpd.DataFrame(df)
        return df
    if "ibis" in name:
        # The Ibis constructor goes through Polars, and Ibis can't read Polars maps.
        return nw.from_native(table).lazy("ibis").to_native()
    return constructor({"a": table["a"]})


def test_get_expr(request: pytest.FixtureRequest, constructor: Constructor) -> None:
    df = nw.from_native(map_frame(request, constructor))
    result = df.select(x=nw.col("a").map.get("x"), y=nw.col("a").map.get("y"))
    assert_equal_data(result, expected)


def test_get_series(
    request: pytest.FixtureRequest, constructor_eager: ConstructorEager
) -> None:
    df = nw.from_native(map_frame(request, constructor_eager), eager_only=True)
    result = {"x": df["a"].map.get("x"), "y": df["a"].map.get("y")}
    assert_equal_data(result, expected)


def test_get_int_key(request: pytest.FixtureRequest, constructor: Constructor) -> None:
    df = nw.from_native(map_frame(request, constructor, int_data))
    result = df.select(x=nw.col("a").map.get(10), y=nw.col("a").map.get(20))
    assert_equal_data(result, expected)


def test_get_null_key(request: pytest.FixtureRequest, constructor: Constructor) -> None:
    df = nw.from_native(map_frame(request, constructor))
    result = df.select(nw.col("a").map.get(None))
    assert_equal_data(result, {"a": [None, None, None, None]})


def test_get_duplicate_keys(
    request: pytest.FixtureRequest, constructor_eager: ConstructorEager
) -> None:
    if "polars" in str(constructor_eager):
        reason = (
            "Polars returns the first value for duplicate keys read from Arrow: "
            "https://github.com/pola-rs/polars/issues/30595"
        )
        request.applymarker(pytest.mark.xfail(reason=reason))
    data = [[("x", 1), ("x", 2)]]
    df = nw.from_native(map_frame(request, constructor_eager, data), eager_only=True)
    assert_equal_data(df.select(nw.col("a").map.get("x")), {"a": [2]})


@pytest.mark.skipif(
    POLARS_VERSION >= PkgVersion("2"),
    reason="native polars.Map dtype is supported on 'polars>=2.0'",
)
def test_get_polars_unsupported() -> None:  # pragma: no cover
    pl = pytest.importorskip("polars")
    df = nw.from_native(pl.DataFrame({"a": [1]}))
    msg = "`Map` dtype is only supported for 'polars>=2.0'"
    with pytest.raises(NotImplementedError, match=re.escape(msg)):
        df.select(nw.col("a").map.get("x"))


def test_get_pandas_non_pyarrow() -> None:
    pd = pytest.importorskip("pandas")
    s = nw.from_native(pd.Series([{"x": 1}]), series_only=True)
    msg = re.escape("Series must be of PyArrow Map type to support map namespace.")
    with pytest.raises(TypeError, match=msg):
        s.map.get("x")


def test_get_dictionary_values(
    request: pytest.FixtureRequest, constructor_eager: ConstructorEager
) -> None:
    pa = pytest.importorskip("pyarrow")
    data = [{"x": "v"}, {"y": "w"}, None]
    value_type = pa.dictionary(pa.int32(), pa.string())
    native = map_frame(request, constructor_eager, data, value_type)
    df = nw.from_native(native, eager_only=True)
    assert_equal_data(df.select(nw.col("a").map.get("x")), {"a": ["v", None, None]})


def test_get_non_map(constructor_eager: ConstructorEager) -> None:
    is_polars = "polars" in str(constructor_eager)
    if is_polars and POLARS_VERSION < PkgVersion("2"):  # pragma: no cover
        pytest.skip(reason="covered by test_get_polars_unsupported")
    df = nw.from_native(constructor_eager({"a": [1, 2]}), eager_only=True)
    with pytest.raises(InvalidOperationError if is_polars else TypeError):
        df.select(nw.col("a").map.get("x"))


@pytest.mark.parametrize(
    ("data", "key"),
    [(int_data, 10.0), (int_data, "10"), (int_data, True), (string_data, 1)],
)
def test_get_wrong_key_type(
    request: pytest.FixtureRequest,
    constructor_eager: ConstructorEager,
    data: list[Any],
    key: Any,
) -> None:
    df = nw.from_native(map_frame(request, constructor_eager, data), eager_only=True)
    with pytest.raises(InvalidOperationError):
        df.select(nw.col("a").map.get(key))


def test_get_naive_key_on_aware_keys() -> None:
    pa = pytest.importorskip("pyarrow")
    utc = dt.timezone.utc
    map_type = pa.map_(pa.timestamp("us", "UTC"), pa.int64())
    array = pa.array([[(dt.datetime(2020, 1, 1, tzinfo=utc), 1)]], type=map_type)
    df = nw.from_native(pa.table({"a": array}))
    assert_equal_data(
        df.select(nw.col("a").map.get(dt.datetime(2020, 1, 1, tzinfo=utc))), {"a": [1]}
    )
    with pytest.raises(InvalidOperationError):
        df.select(nw.col("a").map.get(dt.datetime(2020, 1, 1)))


@pytest.mark.parametrize("key", [2**63, -(2**63) - 1, 2**70])
def test_get_out_of_range_key(
    request: pytest.FixtureRequest, constructor_eager: ConstructorEager, key: int
) -> None:
    df = nw.from_native(map_frame(request, constructor_eager, int_data), eager_only=True)
    assert_equal_data(df.select(nw.col("a").map.get(key)), {"a": [None] * 4})


@pytest.mark.parametrize(
    ("key_type", "stored", "key"),
    [
        (
            "timestamp[s]",
            dt.datetime(2020, 1, 1),
            dt.datetime(2020, 1, 1, 0, 0, 0, 500000),
        ),
        ("uint8", 1, -1),
        ("uint8", 1, 256),
    ],
)
def test_get_unrepresentable_key_pyarrow(key_type: str, stored: Any, key: Any) -> None:
    pa = pytest.importorskip("pyarrow")
    map_type = pa.map_(pa.type_for_alias(key_type), pa.int64())
    array = pa.array([[(stored, 7)]], type=map_type)
    df = nw.from_native(pa.table({"a": array}))
    assert_equal_data(df.select(nw.col("a").map.get(key)), {"a": [None]})
