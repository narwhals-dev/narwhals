# Test assorted functions which we overwrite in stable.v1

from __future__ import annotations

from contextlib import nullcontext as does_not_raise
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import pandas as pd
import pytest

import narwhals as nw
import narwhals.stable.v1 as nw_v1
from narwhals.stable.v1.dependencies import (
    is_cudf_dataframe,
    is_cudf_series,
    is_dask_dataframe,
    is_ibis_table,
    is_modin_dataframe,
    is_modin_series,
    is_pandas_dataframe,
    is_pandas_like_dataframe,
    is_pandas_like_series,
    is_pandas_series,
    is_polars_dataframe,
    is_polars_lazyframe,
    is_polars_series,
    is_pyarrow_chunked_array,
    is_pyarrow_table,
)
from tests.utils import (
    PANDAS_VERSION,
    POLARS_VERSION,
    PYARROW_VERSION,
    Constructor,
    ConstructorEager,
    assert_equal_data,
)

if TYPE_CHECKING:
    from typing_extensions import assert_type

    from tests.utils import Constructor, ConstructorEager


def test_toplevel(constructor_eager: ConstructorEager) -> None:
    if "polars" in str(constructor_eager) and POLARS_VERSION < (1,):
        pytest.skip()
    df = nw_v1.from_native(constructor_eager({"a": [1, 2, 3], "b": [4, 5, 6]}))
    result = df.select(
        min=nw_v1.min("a"),
        max=nw_v1.max("a"),
        mean=nw_v1.mean("a"),
        median=nw_v1.median("a"),
        sum=nw_v1.sum("a"),
        sum_h=nw_v1.sum_horizontal("a"),
        min_h=nw_v1.min_horizontal("a"),
        max_h=nw_v1.max_horizontal("a"),
        mean_h=nw_v1.mean_horizontal("a"),
        len=nw_v1.len(),
        concat_str=nw_v1.concat_str(nw_v1.lit("a"), nw_v1.lit("b")),
        any_h=nw_v1.any_horizontal(nw_v1.lit(True), nw_v1.lit(True)),  # noqa: FBT003
        all_h=nw_v1.all_horizontal(nw_v1.lit(True), nw_v1.lit(True)),  # noqa: FBT003
        first=nw_v1.nth(0),
        no_first=nw_v1.exclude("a"),
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
    }
    assert_equal_data(result, expected)
    assert isinstance(result, nw_v1.DataFrame)


def test_when_then(constructor_eager: ConstructorEager) -> None:
    df = nw_v1.from_native(
        constructor_eager({"a": [1, 2, 3], "b": [4, 5, 6], "c": [6, 7, 8]})
    )
    result = df.select(nw_v1.when(nw_v1.col("a") > 1).then("b").otherwise("c"))
    expected = {"b": [6, 5, 6]}
    assert_equal_data(result, expected)
    assert isinstance(result, nw_v1.DataFrame)


def test_constructors() -> None:
    pytest.importorskip("pyarrow")
    if PANDAS_VERSION < (2, 2):
        pytest.skip()
    assert nw_v1.new_series("a", [1, 2, 3], backend="pandas").to_list() == [1, 2, 3]
    arr: np.ndarray[tuple[int, int], Any] = np.array([[1, 2], [3, 4]])  # pyright: ignore[reportAssignmentType]
    result = nw_v1.from_numpy(arr, schema=["a", "b"], backend="pandas")
    assert_equal_data(result, {"a": [1, 3], "b": [2, 4]})
    assert isinstance(result, nw_v1.DataFrame)
    result = nw_v1.from_numpy(
        arr,
        schema=nw_v1.Schema({"a": nw_v1.Int64(), "b": nw_v1.Int64()}),
        backend="pandas",
    )
    assert_equal_data(result, {"a": [1, 3], "b": [2, 4]})
    assert isinstance(result, nw_v1.DataFrame)
    result = nw_v1.from_dict({"a": [1, 2, 3]}, backend="pandas")
    assert_equal_data(result, {"a": [1, 2, 3]})
    assert isinstance(result, nw_v1.DataFrame)
    result = nw_v1.from_arrow(pd.DataFrame({"a": [1, 2, 3]}), backend="pandas")
    assert_equal_data(result, {"a": [1, 2, 3]})
    assert isinstance(result, nw_v1.DataFrame)


def test_join(constructor_eager: ConstructorEager) -> None:
    df = nw_v1.from_native(constructor_eager({"a": [1, 2, 3]})).lazy()
    result = df.join(df, how="inner", on="a").sort("a")  # type: ignore[arg-type]
    expected = {"a": [1, 2, 3]}
    assert_equal_data(result, expected)
    assert isinstance(result, nw_v1.LazyFrame)
    result_eager = df.collect().join(df.collect(), how="inner", on="a")
    assert_equal_data(result_eager, expected)
    assert isinstance(result_eager, nw_v1.DataFrame)


def test_by_name(constructor_eager: ConstructorEager) -> None:
    df = nw_v1.from_native(constructor_eager({"a": [1, 2, 3]})).lazy()
    result = df.select(nw_v1.col("a").alias("b"), "a")
    expected = {"b": [1, 2, 3], "a": [1, 2, 3]}
    assert_equal_data(result, expected)
    assert isinstance(result, nw_v1.LazyFrame)
    result_eager = df.collect().select(nw_v1.col("a").alias("b"), "a")
    assert_equal_data(result_eager, expected)
    assert isinstance(result_eager, nw_v1.DataFrame)


def test_values_counts_v1(constructor_eager: ConstructorEager) -> None:
    df = nw_v1.from_native(constructor_eager({"a": [1, 2, 3]}), eager_only=True)
    result = df["a"].value_counts().sort("a")
    expected = {"a": [1, 2, 3], "count": [1, 1, 1]}
    assert_equal_data(result, expected)
    assert isinstance(result, nw_v1.DataFrame)


def test_is_duplicated_unique(constructor_eager: ConstructorEager) -> None:
    df = nw_v1.from_native(constructor_eager({"a": [1, 2, 3]}), eager_only=True)
    assert df.is_duplicated().to_list() == [False, False, False]
    assert df.is_unique().to_list() == [True, True, True]
    assert isinstance(df.is_duplicated(), nw_v1.Series)
    assert isinstance(df.is_unique(), nw_v1.Series)


def test_concat(constructor_eager: ConstructorEager) -> None:
    df = nw_v1.from_native(constructor_eager({"a": [1, 2, 3]}), eager_only=True)
    result = nw_v1.concat([df, df], how="vertical")
    expected = {"a": [1, 2, 3, 1, 2, 3]}
    assert_equal_data(result, expected)
    assert isinstance(result, nw_v1.DataFrame)
    if TYPE_CHECKING:
        assert_type(result, nw_v1.DataFrame[Any])


@pytest.mark.filterwarnings(
    "ignore:.*all arguments of to_dict except for the argument:FutureWarning"
)
def test_to_dict(constructor_eager: ConstructorEager) -> None:
    df = nw_v1.from_native(constructor_eager({"a": [1, 2, 3]}), eager_only=True)
    result = df.to_dict(as_series=False)
    expected = {"a": [1, 2, 3]}
    assert result == expected


def test_tail(constructor_eager: ConstructorEager) -> None:
    df = nw_v1.from_native(constructor_eager({"a": [1, 2, 3]}), eager_only=True).lazy()
    result = df.tail(3)
    expected = {"a": [1, 2, 3]}
    assert_equal_data(result, expected)


@pytest.mark.filterwarnings(
    "ignore:.*all arguments of to_dict except for the argument:FutureWarning"
)
def test_to_dict_as_series(constructor_eager: ConstructorEager) -> None:
    df = nw_v1.from_native(constructor_eager({"a": [1, 2, 3]}), eager_only=True)
    result = df.to_dict(as_series=True)
    expected = {"a": [1, 2, 3]}
    assert_equal_data(result, expected)
    assert isinstance(result["a"], nw_v1.Series)


@pytest.mark.filterwarnings(
    "ignore:`Series.hist` is being called from the stable API although considered an unstable feature."
)
def test_hist_v1(constructor_eager: ConstructorEager) -> None:
    if "pyarrow_table" in str(constructor_eager) and PYARROW_VERSION < (13,):
        pytest.skip()
    if "cudf" in str(constructor_eager):
        pytest.skip()
    df = nw_v1.from_native(constructor_eager({"a": [1, 1, 2]}), eager_only=True)
    result = df["a"].hist(bins=[-1, 1, 2])
    expected = {"breakpoint": [1, 2], "count": [2, 1]}
    assert_equal_data(result, expected)
    assert isinstance(result, nw_v1.DataFrame)


@pytest.mark.skipif(PANDAS_VERSION < (2, 0), reason="requires interchange protocol")
def test_is_ordered_categorical_interchange_protocol() -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    df = pd.DataFrame(
        {"a": ["a", "b"]}, dtype=pd.CategoricalDtype(ordered=True)
    ).__dataframe__()
    assert nw_v1.is_ordered_categorical(
        nw_v1.from_native(df, eager_or_interchange_only=True)["a"]
    )


def test_all_nulls_pandas() -> None:
    assert (
        nw_v1.from_native(pd.Series([None] * 3, dtype="object"), series_only=True).dtype
        == nw_v1.Object
    )


def test_int_select_pandas() -> None:
    df = nw_v1.from_native(pd.DataFrame({0: [1, 2], "b": [3, 4]}))
    with pytest.raises(
        nw_v1.exceptions.InvalidIntoExprError, match="\n\nHint:\n- if you were trying"
    ):
        nw_v1.to_native(df.select(0))  # type: ignore[arg-type]
    with pytest.raises(
        nw_v1.exceptions.InvalidIntoExprError, match="\n\nHint:\n- if you were trying"
    ):
        nw_v1.to_native(df.lazy().select(0))  # type: ignore[arg-type]


def test_enum_v1_is_enum_unstable() -> None:
    enum_v1 = nw_v1.Enum()
    enum_unstable = nw.Enum(("a", "b", "c"))
    assert isinstance(enum_v1, nw.Enum)
    assert issubclass(nw_v1.Enum, nw.Enum)
    assert enum_v1 == nw.Enum
    assert enum_v1 != enum_unstable
    assert enum_unstable != nw_v1.Enum
    assert enum_unstable == nw.Enum

    with pytest.raises(TypeError, match=r"takes 1 positional argument"):
        nw_v1.Enum(("a", "b"))  # type: ignore[call-arg]


def test_cast_to_enum_v1(
    request: pytest.FixtureRequest, constructor: Constructor
) -> None:
    # Backends that do not (yet) support Enum dtype
    if any(
        backend in str(constructor)
        for backend in ("pyarrow_table", "sqlframe", "pyspark", "ibis")
    ):
        request.applymarker(pytest.mark.xfail)

    df_native = constructor({"a": ["a", "b"]})

    with pytest.raises(
        NotImplementedError,
        match="Converting to Enum is not supported in narwhals.stable.v1",
    ):
        nw_v1.from_native(df_native).select(nw_v1.col("a").cast(nw_v1.Enum))  # type: ignore[arg-type]


def test_v1_ordered_categorical_pandas() -> None:
    s = nw_v1.from_native(
        pd.Series([0, 1], dtype=pd.CategoricalDtype(ordered=True)), series_only=True
    )
    assert s.dtype == nw_v1.Categorical


def test_v1_enum_polars() -> None:
    pytest.importorskip("polars")
    import polars as pl

    s = nw_v1.from_native(
        pl.Series(["a", "b"], dtype=pl.Enum(["a", "b"])), series_only=True
    )
    assert s.dtype == nw_v1.Enum


def test_v1_enum_duckdb_2550() -> None:
    pytest.importorskip("duckdb")
    import duckdb

    result_v1 = nw_v1.from_native(
        duckdb.sql("select 'a'::enum('a', 'b', 'c') as a")
    ).collect_schema()
    assert result_v1 == {"a": nw_v1.Enum()}
    result = nw.from_native(
        duckdb.sql("select 'a'::enum('a', 'b', 'c') as a")
    ).collect_schema()
    assert result == {"a": nw.Enum(("a", "b", "c"))}


@pytest.mark.parametrize(
    "is_native_dataframe",
    [
        is_pandas_dataframe,
        is_dask_dataframe,
        is_modin_dataframe,
        is_polars_dataframe,
        is_cudf_dataframe,
        is_ibis_table,
        is_polars_lazyframe,
        is_pyarrow_table,
        is_pandas_like_dataframe,
    ],
)
def test_is_native_dataframe(is_native_dataframe: Callable[[Any], Any]) -> None:
    data = {"a": [1, 2], "b": ["bar", "foo"]}
    df = nw.from_native(pd.DataFrame(data))
    assert not is_native_dataframe(df)


@pytest.mark.parametrize(
    "is_native_series",
    [
        is_pandas_series,
        is_modin_series,
        is_polars_series,
        is_cudf_series,
        is_pyarrow_chunked_array,
        is_pandas_like_series,
    ],
)
def test_is_native_series(is_native_series: Callable[[Any], Any]) -> None:
    data = {"a": [1, 2]}
    ser = nw.from_native(pd.DataFrame(data))["a"]
    assert not is_native_series(ser)


def test_get_level() -> None:
    pytest.importorskip("polars")
    import polars as pl

    df = pl.DataFrame({"a": [1, 2, 3]})
    with pytest.deprecated_call():
        assert nw.get_level(nw.from_native(df)) == "full"
    assert nw_v1.get_level(nw_v1.from_native(df)) == "full"
    assert (
        nw_v1.get_level(
            nw_v1.from_native(df.__dataframe__(), eager_or_interchange_only=True)
        )
        == "interchange"
    )


def test_any_horizontal() -> None:
    # here, it defaults to Kleene logic.
    pytest.importorskip("polars")
    import polars as pl

    df = nw_v1.from_native(
        pl.DataFrame({"a": [True, True, False], "b": [True, None, None]})
    )
    result = df.select(nw_v1.any_horizontal("a", "b"))
    expected = {"a": [True, True, None]}
    assert_equal_data(result, expected)
    with pytest.deprecated_call(match="ignore_nulls"):
        result = df.select(nw.any_horizontal("a", "b"))
    assert_equal_data(result, expected)


def test_all_horizontal() -> None:
    # here, it defaults to Kleene logic.
    pytest.importorskip("polars")
    import polars as pl

    df = nw_v1.from_native(
        pl.DataFrame({"a": [True, True, False], "b": [True, None, None]})
    )
    result = df.select(nw_v1.all_horizontal("a", "b"))
    expected = {"a": [True, None, False]}
    assert_equal_data(result, expected)
    with pytest.deprecated_call(match="ignore_nulls"):
        result = df.select(nw.all_horizontal("a", "b"))
    assert_equal_data(result, expected)


def test_with_row_index(constructor: Constructor) -> None:
    data = {"abc": ["foo", "bars"], "xyz": [100, 200], "const": [42, 42]}

    frame = nw_v1.from_native(constructor(data))

    msg = "Cannot pass `order_by`"
    context = (
        pytest.raises(TypeError, match=msg)
        if any(x in str(constructor) for x in ("duckdb", "pyspark"))
        else does_not_raise()
    )

    with context:
        result = frame.with_row_index()

        expected = {"index": [0, 1], **data}
        assert_equal_data(result, expected)
