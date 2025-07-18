# Test assorted functions which we overwrite in stable.v2

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest

import narwhals.stable.v2 as nw_v2
from narwhals.utils import Version
from tests.utils import PANDAS_VERSION, Constructor, assert_equal_data

if TYPE_CHECKING:
    from typing_extensions import assert_type

    from narwhals.stable.v2.typing import IntoDataFrameT


def test_toplevel() -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    df = nw_v2.from_native(
        pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, None, 9]})
    )
    result = df.select(
        min=nw_v2.min("a"),
        max=nw_v2.max("a"),
        mean=nw_v2.mean("a"),
        median=nw_v2.median("a"),
        sum=nw_v2.sum("a"),
        sum_h=nw_v2.sum_horizontal("a"),
        min_h=nw_v2.min_horizontal("a"),
        max_h=nw_v2.max_horizontal("a"),
        mean_h=nw_v2.mean_horizontal("a"),
        len=nw_v2.len(),
        concat_str=nw_v2.concat_str(nw_v2.lit("a"), nw_v2.lit("b")),
        any_h=nw_v2.any_horizontal(nw_v2.lit(True), nw_v2.lit(True), ignore_nulls=True),  # noqa: FBT003
        all_h=nw_v2.all_horizontal(nw_v2.lit(True), nw_v2.lit(True), ignore_nulls=True),  # noqa: FBT003
        first=nw_v2.nth(0),
        no_first=nw_v2.exclude("a", "c"),
        coalesce=nw_v2.coalesce("c", "a"),
    )
    expected = {
        "min": [1, 1, 1],
        "max": [3, 3, 3],
        "mean": [2.0, 2.0, 2.0],
        "median": [2.0, 2.0, 2.0],
        "sum": [6, 6, 6],
        "sum_h": [1, 2, 3],
        "min_h": [1, 2, 3],
        "max_h": [1, 2, 3],
        "mean_h": [1, 2, 3],
        "len": [3, 3, 3],
        "concat_str": ["ab", "ab", "ab"],
        "any_h": [True, True, True],
        "all_h": [True, True, True],
        "first": [1, 2, 3],
        "no_first": [4, 5, 6],
        "coalesce": [7, 2, 9],
    }
    assert_equal_data(result, expected)
    assert isinstance(result, nw_v2.DataFrame)


def test_when_then() -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    df = nw_v2.from_native(pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [6, 7, 8]}))
    result = df.select(nw_v2.when(nw_v2.col("a") > 1).then("b").otherwise("c"))
    expected = {"b": [6, 5, 6]}
    assert_equal_data(result, expected)
    assert isinstance(result, nw_v2.DataFrame)


def test_constructors() -> None:
    pytest.importorskip("pyarrow")
    if PANDAS_VERSION < (2, 2):
        pytest.skip()
    assert nw_v2.new_series("a", [1, 2, 3], backend="pandas").to_list() == [1, 2, 3]
    arr: np.ndarray[tuple[int, int], Any] = np.array([[1, 2], [3, 4]])  # pyright: ignore[reportAssignmentType]
    result = nw_v2.from_numpy(arr, schema=["a", "b"], backend="pandas")
    assert_equal_data(result, {"a": [1, 3], "b": [2, 4]})
    assert isinstance(result, nw_v2.DataFrame)
    result = nw_v2.from_numpy(
        arr,
        schema=nw_v2.Schema({"a": nw_v2.Int64(), "b": nw_v2.Int64()}),
        backend="pandas",
    )
    assert_equal_data(result, {"a": [1, 3], "b": [2, 4]})
    assert isinstance(result, nw_v2.DataFrame)
    result = nw_v2.from_dict({"a": [1, 2, 3]}, backend="pandas")
    assert_equal_data(result, {"a": [1, 2, 3]})
    assert isinstance(result, nw_v2.DataFrame)
    result = nw_v2.from_arrow(pd.DataFrame({"a": [1, 2, 3]}), backend="pandas")
    assert_equal_data(result, {"a": [1, 2, 3]})
    assert isinstance(result, nw_v2.DataFrame)


def test_values_counts_v2() -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    df = nw_v2.from_native(pd.DataFrame({"a": [1, 2, 3]}), eager_only=True)
    result = df["a"].value_counts().sort("a")
    expected = {"a": [1, 2, 3], "count": [1, 1, 1]}
    assert_equal_data(result, expected)
    assert isinstance(result, nw_v2.DataFrame)


def test_is_duplicated_unique() -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    df = nw_v2.from_native(pd.DataFrame({"a": [1, 2, 3]}), eager_only=True)
    assert df.is_duplicated().to_list() == [False, False, False]
    assert df.is_unique().to_list() == [True, True, True]
    assert isinstance(df.is_duplicated(), nw_v2.Series)
    assert isinstance(df.is_unique(), nw_v2.Series)


def test_concat() -> None:
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    df = nw_v2.from_native(pa.table({"a": [1, 2, 3]}), eager_only=True)
    result = nw_v2.concat([df, df], how="vertical")
    expected = {"a": [1, 2, 3, 1, 2, 3]}
    assert_equal_data(result, expected)
    assert isinstance(result, nw_v2.DataFrame)
    if TYPE_CHECKING:
        assert_type(result, nw_v2.DataFrame[Any])


def test_to_dict_as_series() -> None:
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    df = nw_v2.from_native(pa.table({"a": [1, 2, 3]}), eager_only=True)
    result = df.to_dict(as_series=True)
    expected = {"a": [1, 2, 3]}
    assert_equal_data(result, expected)
    assert isinstance(result["a"], nw_v2.Series)


def test_from_native_already_nw() -> None:
    pytest.importorskip("polars")
    import polars as pl

    df = nw_v2.from_native(pl.DataFrame({"a": [1]}))
    assert isinstance(nw_v2.from_native(df), nw_v2.DataFrame)
    assert nw_v2.from_native(df) is df
    lf = nw_v2.from_native(pl.LazyFrame({"a": [1]}))
    assert isinstance(nw_v2.from_native(lf), nw_v2.LazyFrame)
    assert nw_v2.from_native(lf) is lf
    s = df["a"]
    assert isinstance(nw_v2.from_native(s, series_only=True), nw_v2.Series)
    assert nw_v2.from_native(df) is df


def test_from_native_invalid_kwds() -> None:
    pytest.importorskip("polars")
    import polars as pl

    with pytest.raises(TypeError, match="got an unexpected keyword"):
        nw_v2.from_native(pl.DataFrame({"a": [1]}), belugas=True)  # type: ignore[call-overload]


def test_io(tmpdir: pytest.TempdirFactory) -> None:
    pytest.importorskip("polars")
    import polars as pl

    csv_filepath = str(tmpdir / "file.csv")  # type: ignore[operator]
    parquet_filepath = str(tmpdir / "file.parquet")  # type: ignore[operator]
    pl.DataFrame({"a": [1]}).write_csv(csv_filepath)
    pl.DataFrame({"a": [1]}).write_parquet(parquet_filepath)
    assert isinstance(nw_v2.read_csv(csv_filepath, backend="polars"), nw_v2.DataFrame)
    assert isinstance(nw_v2.scan_csv(csv_filepath, backend="polars"), nw_v2.LazyFrame)
    assert isinstance(
        nw_v2.read_parquet(parquet_filepath, backend="polars"), nw_v2.DataFrame
    )
    assert isinstance(
        nw_v2.scan_parquet(parquet_filepath, backend="polars"), nw_v2.LazyFrame
    )


def test_narwhalify() -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    data = {"a": [2, 3, 4]}

    @nw_v2.narwhalify
    def func(df: nw_v2.DataFrame[IntoDataFrameT]) -> nw_v2.DataFrame[IntoDataFrameT]:
        return df.with_columns(nw_v2.all() + 1)

    df = pd.DataFrame({"a": [1, 2, 3]})
    result = func(df)
    pd.testing.assert_frame_equal(result, pd.DataFrame(data))
    result = func(df=df)
    pd.testing.assert_frame_equal(result, pd.DataFrame(data))


def test_narwhalify_method() -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    data = {"a": [2, 3, 4]}

    class Foo:
        @nw_v2.narwhalify
        def func(
            self, df: nw_v2.DataFrame[IntoDataFrameT], a: int = 1
        ) -> nw_v2.DataFrame[IntoDataFrameT]:
            return df.with_columns(nw_v2.all() + a)

    df = pd.DataFrame({"a": [1, 2, 3]})
    result = Foo().func(df)
    pd.testing.assert_frame_equal(result, pd.DataFrame(data))
    result = Foo().func(a=1, df=df)
    pd.testing.assert_frame_equal(result, pd.DataFrame(data))


def test_narwhalify_method_called() -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    data = {"a": [2, 3, 4]}

    class Foo:
        @nw_v2.narwhalify
        def func(
            self, df: nw_v2.DataFrame[IntoDataFrameT], a: int = 1
        ) -> nw_v2.DataFrame[IntoDataFrameT]:
            return df.with_columns(nw_v2.all() + a)

    df = pd.DataFrame({"a": [1, 2, 3]})
    result = Foo().func(df)
    pd.testing.assert_frame_equal(result, pd.DataFrame(data))
    result = Foo().func(df=df)
    pd.testing.assert_frame_equal(result, pd.DataFrame(data))
    result = Foo().func(a=1, df=df)
    pd.testing.assert_frame_equal(result, pd.DataFrame(data))


def test_with_version(constructor: Constructor) -> None:
    lf = nw_v2.from_native(constructor({"a": [1, 2]})).lazy()
    assert isinstance(lf, nw_v2.LazyFrame)
    assert lf._compliant_frame._with_version(Version.MAIN)._version is Version.MAIN
