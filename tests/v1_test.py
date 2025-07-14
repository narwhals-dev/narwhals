# Test assorted functions which we overwrite in stable.v1
from __future__ import annotations

from contextlib import nullcontext as does_not_raise
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import pandas as pd
import pytest

import narwhals as nw
import narwhals.stable.v1 as nw_v1
from narwhals.exceptions import InvalidOperationError
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
from narwhals.utils import Version
from tests.utils import (
    DUCKDB_VERSION,
    PANDAS_VERSION,
    POLARS_VERSION,
    Constructor,
    ConstructorEager,
    assert_equal_data,
)

if TYPE_CHECKING:
    from typing_extensions import assert_type

    from narwhals.typing import IntoDataFrameT
    from tests.utils import Constructor, ConstructorEager


def test_toplevel(constructor_eager: ConstructorEager) -> None:
    if "polars" in str(constructor_eager) and POLARS_VERSION < (1,):
        pytest.skip()
    df = nw_v1.from_native(
        constructor_eager({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, None, 9]})
    )
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
        no_first=nw_v1.exclude("a", "c"),
        coalesce=nw_v1.coalesce("c", "a"),
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


def test_renamed_taxicab_norm(constructor: Constructor) -> None:
    if "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 3):
        pytest.skip()
    # Suppose we need to rename `_l1_norm` to `_taxicab_norm`.
    # We need `narwhals.stable.v1` to stay stable. So, we
    # make the change in `narwhals`, and then add the new method
    # to the subclass of `Expr` in `narwhals.stable.v1`.
    # Here, we check that anyone who wrote code using the old
    # API will still be able to use it, without the main namespace
    # getting cluttered by the new name.

    df = nw.from_native(constructor({"a": [1, 2, 3, -4, 5]}))
    result = df.with_columns(b=nw.col("a")._taxicab_norm())
    expected = {"a": [1, 2, 3, -4, 5], "b": [15] * 5}
    assert_equal_data(result, expected)

    with pytest.raises(AttributeError):
        result = df.with_columns(b=nw.col("a")._l1_norm())  # type: ignore[attr-defined]

    df_v1 = nw_v1.from_native(constructor({"a": [1, 2, 3, -4, 5]}))
    # The newer `_taxicab_norm` can still work in the old API, no issue.
    # It's new, so it couldn't be backwards-incompatible.
    result_v1 = df_v1.with_columns(b=nw_v1.col("a")._taxicab_norm())
    expected = {"a": [1, 2, 3, -4, 5], "b": [15] * 5}
    assert_equal_data(result_v1, expected)

    # The older `_l1_norm` still works in the stable api
    result_v1 = df_v1.with_columns(b=nw_v1.col("a")._l1_norm())
    assert_equal_data(result_v1, expected)


def test_renamed_taxicab_norm_dataframe(constructor: Constructor) -> None:
    # Suppose we have `DataFrame._l1_norm` in `stable.v1`, but remove it
    # in the main namespace. Here, we check that it's still usable from
    # the stable api.
    def func(df_any: Any) -> Any:
        df = nw_v1.from_native(df_any)
        df = df._l1_norm()
        return df.to_native()

    result = nw_v1.from_native(func(constructor({"a": [1, 2, 3, -4, 5]})))
    expected = {"a": [15]}
    assert_equal_data(result, expected)


def test_renamed_taxicab_norm_dataframe_narwhalify(constructor: Constructor) -> None:
    # Suppose we have `DataFrame._l1_norm` in `stable.v1`, but remove it
    # in the main namespace. Here, we check that it's still usable from
    # the stable api when using `narwhalify`.
    @nw_v1.narwhalify
    def func(df: Any) -> Any:
        return df._l1_norm()

    result = nw_v1.from_native(func(constructor({"a": [1, 2, 3, -4, 5]})))
    expected = {"a": [15]}
    assert_equal_data(result, expected)


def test_dtypes() -> None:
    df = nw_v1.from_native(
        pd.DataFrame({"a": [1], "b": [datetime(2020, 1, 1)], "c": [timedelta(1)]})
    )
    dtype = df.collect_schema()["b"]
    assert dtype in {nw_v1.Datetime}
    assert isinstance(dtype, nw_v1.Datetime)
    dtype = df.collect_schema()["c"]
    assert dtype in {nw_v1.Duration}
    assert isinstance(dtype, nw_v1.Duration)


@pytest.mark.parametrize(
    ("strict", "context"),
    [
        (
            True,
            pytest.raises(
                TypeError,
                match="Expected pandas-like dataframe, Polars dataframe, or Polars lazyframe",
            ),
        ),
        (False, does_not_raise()),
    ],
)
def test_strict(strict: Any, context: Any) -> None:
    arr = np.array([1, 2, 3])

    with context:
        res = nw_v1.from_native(arr, strict=strict)
        assert isinstance(res, np.ndarray)


def test_from_native_strict_false_typing() -> None:
    pytest.importorskip("polars")
    import polars as pl

    df = pl.DataFrame()
    nw_v1.from_native(df, strict=False)
    nw_v1.from_native(df, strict=False, eager_only=True)
    nw_v1.from_native(df, strict=False, eager_or_interchange_only=True)

    with pytest.deprecated_call(match="please use `pass_through` instead"):
        nw.from_native(df, strict=False)  # type: ignore[call-overload]
        nw.from_native(df, strict=False, eager_only=True)  # type: ignore[call-overload]


def test_from_native_strict_false_invalid() -> None:
    with pytest.raises(ValueError, match="Cannot pass both `strict`"):
        nw_v1.from_native({"a": [1, 2, 3]}, strict=True, pass_through=False)  # type: ignore[call-overload]


def test_from_mock_interchange_protocol_non_strict() -> None:
    class MockDf:
        def __dataframe__(self) -> None:  # pragma: no cover
            pass

    mockdf = MockDf()
    result = nw_v1.from_native(mockdf, eager_only=True, strict=False)
    assert result is mockdf


def test_from_native_lazyframe() -> None:
    pytest.importorskip("polars")
    import polars as pl

    stable_lazy = nw_v1.from_native(pl.LazyFrame({"a": [1]}))
    if TYPE_CHECKING:
        assert_type(stable_lazy, nw_v1.LazyFrame[pl.LazyFrame])

    assert isinstance(stable_lazy, nw_v1.LazyFrame)


def test_dataframe_recursive_v1() -> None:
    """`v1` always returns a `Union` for `DataFrame`."""
    pytest.importorskip("polars")
    import polars as pl

    pl_frame = pl.DataFrame({"a": [1, 2, 3]})
    nw_frame = nw_v1.from_native(pl_frame)
    with pytest.raises(AttributeError):
        nw_v1.DataFrame(nw_frame, level="full")

    nw_frame_early_return = nw_v1.from_native(nw_frame)

    if TYPE_CHECKING:
        assert_type(pl_frame, pl.DataFrame)
        assert_type(
            nw_frame, "nw_v1.DataFrame[pl.DataFrame] | nw_v1.LazyFrame[pl.DataFrame]"
        )
        nw_frame_depth_2 = nw_v1.DataFrame(nw_frame, level="full")  # type: ignore[var-annotated]
        assert_type(nw_frame_depth_2, nw_v1.DataFrame[Any])
        # NOTE: Checking that the type is `DataFrame[Unknown]`
        assert_type(
            nw_frame_early_return,
            "nw_v1.DataFrame[pl.DataFrame] | nw_v1.LazyFrame[pl.DataFrame]",
        )


def test_lazyframe_recursive_v1() -> None:
    pytest.importorskip("polars")
    import polars as pl

    pl_frame = pl.DataFrame({"a": [1, 2, 3]}).lazy()
    nw_frame = nw_v1.from_native(pl_frame)
    with pytest.raises(AttributeError):
        nw_v1.LazyFrame(nw_frame, level="lazy")

    nw_frame_early_return = nw_v1.from_native(nw_frame)

    if TYPE_CHECKING:
        assert_type(pl_frame, pl.LazyFrame)
        assert_type(nw_frame, nw_v1.LazyFrame[pl.LazyFrame])

        nw_frame_depth_2 = nw_v1.LazyFrame(nw_frame, level="lazy")  # type: ignore[var-annotated]
        # NOTE: Checking that the type is `LazyFrame[Unknown]`
        assert_type(nw_frame_depth_2, nw_v1.LazyFrame[Any])
        assert_type(nw_frame_early_return, nw_v1.LazyFrame[pl.LazyFrame])


def test_series_recursive_v1() -> None:
    """https://github.com/narwhals-dev/narwhals/issues/2239."""
    pytest.importorskip("polars")
    import polars as pl

    pl_series = pl.Series(name="test", values=[1, 2, 3])
    nw_series = nw_v1.from_native(pl_series, series_only=True)
    with pytest.raises(AttributeError):
        nw_v1.Series(nw_series, level="full")

    nw_series_early_return = nw_v1.from_native(nw_series, series_only=True)

    if TYPE_CHECKING:
        assert_type(pl_series, pl.Series)
        assert_type(nw_series, nw_v1.Series[pl.Series])

        nw_series_depth_2 = nw_v1.Series(nw_series, level="full")
        # NOTE: `Unknown` isn't possible for `v1`, as it has a `TypeVar` default
        assert_type(nw_series_depth_2, nw_v1.Series[Any])
        assert_type(nw_series_early_return, nw_v1.Series[pl.Series])


def test_from_native_already_nw() -> None:
    pytest.importorskip("polars")
    import polars as pl

    df = nw_v1.from_native(pl.DataFrame({"a": [1]}))
    assert isinstance(nw_v1.from_native(df), nw_v1.DataFrame)
    assert nw_v1.from_native(df) is df
    lf = nw_v1.from_native(pl.LazyFrame({"a": [1]}))
    assert isinstance(nw_v1.from_native(lf), nw_v1.LazyFrame)
    assert nw_v1.from_native(lf) is lf
    s = df["a"]
    assert isinstance(nw_v1.from_native(s, series_only=True), nw_v1.Series)
    assert nw_v1.from_native(df) is df


def test_from_native_invalid_kwds() -> None:
    pytest.importorskip("polars")
    import polars as pl

    with pytest.raises(TypeError, match="got an unexpected keyword"):
        nw_v1.from_native(pl.DataFrame({"a": [1]}), belugas=True)  # type: ignore[call-overload]


def test_io(tmpdir: pytest.TempdirFactory) -> None:
    pytest.importorskip("polars")
    import polars as pl

    csv_filepath = str(tmpdir / "file.csv")  # type: ignore[operator]
    parquet_filepath = str(tmpdir / "file.parquet")  # type: ignore[operator]
    pl.DataFrame({"a": [1]}).write_csv(csv_filepath)
    pl.DataFrame({"a": [1]}).write_parquet(parquet_filepath)
    assert isinstance(nw_v1.read_csv(csv_filepath, backend="polars"), nw_v1.DataFrame)
    assert isinstance(nw_v1.scan_csv(csv_filepath, backend="polars"), nw_v1.LazyFrame)
    assert isinstance(
        nw_v1.read_parquet(parquet_filepath, backend="polars"), nw_v1.DataFrame
    )
    assert isinstance(
        nw_v1.scan_parquet(parquet_filepath, backend="polars"), nw_v1.LazyFrame
    )


def test_narwhalify() -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    data = {"a": [2, 3, 4]}

    @nw_v1.narwhalify
    def func(df: nw_v1.DataFrame[IntoDataFrameT]) -> nw_v1.DataFrame[IntoDataFrameT]:
        return df.with_columns(nw_v1.all() + 1)

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
        @nw_v1.narwhalify
        def func(
            self, df: nw_v1.DataFrame[IntoDataFrameT], a: int = 1
        ) -> nw_v1.DataFrame[IntoDataFrameT]:
            return df.with_columns(nw_v1.all() + a)

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
        @nw_v1.narwhalify
        def func(
            self, df: nw_v1.DataFrame[IntoDataFrameT], a: int = 1
        ) -> nw_v1.DataFrame[IntoDataFrameT]:
            return df.with_columns(nw_v1.all() + a)

    df = pd.DataFrame({"a": [1, 2, 3]})
    result = Foo().func(df)
    pd.testing.assert_frame_equal(result, pd.DataFrame(data))
    result = Foo().func(df=df)
    pd.testing.assert_frame_equal(result, pd.DataFrame(data))
    result = Foo().func(a=1, df=df)
    pd.testing.assert_frame_equal(result, pd.DataFrame(data))


def test_narwhalify_method_invalid() -> None:
    class Foo:
        @nw_v1.narwhalify(strict=True, eager_only=True)
        def func(self) -> Foo:  # pragma: no cover
            return self

        @nw_v1.narwhalify(strict=True, eager_only=True)
        def fun2(self, df: Any) -> Any:  # pragma: no cover
            return df

    with pytest.raises(TypeError):
        Foo().func()


def test_narwhalify_invalid() -> None:
    @nw_v1.narwhalify(strict=True)
    def func() -> None:  # pragma: no cover
        return None

    with pytest.raises(TypeError):
        func()


def test_narwhalify_backends_pandas() -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    data = {"a": [2, 3, 4]}

    @nw_v1.narwhalify
    def func(
        arg1: Any, arg2: Any, extra: int = 1
    ) -> tuple[Any, Any, int]:  # pragma: no cover
        return arg1, arg2, extra

    func(pd.DataFrame(data), pd.Series(data["a"]))


def test_narwhalify_backends_polars() -> None:
    pytest.importorskip("polars")
    import polars as pl

    data = {"a": [2, 3, 4]}

    @nw_v1.narwhalify
    def func(
        arg1: Any, arg2: Any, extra: int = 1
    ) -> tuple[Any, Any, int]:  # pragma: no cover
        return arg1, arg2, extra

    func(pl.DataFrame(data), pl.Series(data["a"]))


def test_narwhalify_backends_cross() -> None:
    pytest.importorskip("pandas")
    pytest.importorskip("polars")
    import pandas as pd
    import polars as pl

    data = {"a": [2, 3, 4]}

    @nw_v1.narwhalify
    def func(
        arg1: Any, arg2: Any, extra: int = 1
    ) -> tuple[Any, Any, int]:  # pragma: no cover
        return arg1, arg2, extra

    with pytest.raises(
        ValueError,
        match="Found multiple backends. Make sure that all dataframe/series inputs come from the same backend.",
    ):
        func(pd.DataFrame(data), pl.DataFrame(data))


def test_narwhalify_backends_cross2() -> None:
    pytest.importorskip("pandas")
    pytest.importorskip("polars")
    import pandas as pd
    import polars as pl

    data = {"a": [2, 3, 4]}

    @nw_v1.narwhalify
    def func(
        arg1: Any, arg2: Any, extra: int = 1
    ) -> tuple[Any, Any, int]:  # pragma: no cover
        return arg1, arg2, extra

    with pytest.raises(
        ValueError,
        match="Found multiple backends. Make sure that all dataframe/series inputs come from the same backend.",
    ):
        func(pl.DataFrame(data), pd.Series(data["a"]))


def test_expr_sample(constructor_eager: ConstructorEager) -> None:
    df = nw_v1.from_native(
        constructor_eager({"a": [1, 2, 3], "b": [4, 5, 6]}), eager_only=True
    )

    result_expr = df.select(nw_v1.col("a").sample(n=2)).shape
    expected_expr = (2, 1)
    assert result_expr == expected_expr

    with pytest.deprecated_call(
        match="is deprecated and will be removed in a future version"
    ):
        df.select(nw.col("a").sample(n=2))


def test_is_frame(constructor: Constructor) -> None:
    lf = nw_v1.from_native(constructor({"a": [1, 2]})).lazy()
    assert isinstance(lf, nw_v1.LazyFrame)
    assert nw_v1.dependencies.is_narwhals_lazyframe(lf)
    assert nw_v1.dependencies.is_narwhals_dataframe(lf.collect())


def test_with_version(constructor: Constructor) -> None:
    lf = nw_v1.from_native(constructor({"a": [1, 2]})).lazy()
    assert isinstance(lf, nw_v1.LazyFrame)
    assert lf._compliant_frame._with_version(Version.MAIN)._version is Version.MAIN


@pytest.mark.parametrize("n", [1, 2])
@pytest.mark.parametrize("offset", [1, 2])
def test_gather_every(constructor_eager: ConstructorEager, n: int, offset: int) -> None:
    data = {"a": list(range(10))}
    df_v1 = nw_v1.from_native(constructor_eager(data))
    result = df_v1.gather_every(n=n, offset=offset)
    expected = {"a": data["a"][offset::n]}
    assert_equal_data(result, expected)

    # Test deprecation for LazyFrame in main namespace
    lf = nw.from_native(constructor_eager(data)).lazy()
    with pytest.deprecated_call(
        match="is deprecated and will be removed in a future version"
    ):
        lf.gather_every(n=n, offset=offset)


@pytest.mark.parametrize("n", [1, 2])
@pytest.mark.parametrize("offset", [1, 2])
def test_gather_every_dask_v1(n: int, offset: int) -> None:
    pytest.importorskip("dask")
    import dask.dataframe as dd

    data = {"a": list(range(10))}

    df_v1 = nw_v1.from_native(dd.from_pandas(pd.DataFrame(data)))
    result = df_v1.gather_every(n=n, offset=offset)
    expected = {"a": data["a"][offset::n]}
    assert_equal_data(result, expected)


def test_unique_series_v1() -> None:
    pytest.importorskip("polars")
    import polars as pl

    data = {"a": [1, 1, 2]}
    series = nw.from_native(pl.DataFrame(data), eager_only=True)["a"]
    # this shouldn't warn
    series.to_frame().select(nw_v1.col("a").unique().sum())

    series = nw_v1.from_native(pl.DataFrame(data), eager_only=True)["a"]
    with pytest.warns(
        UserWarning,
        match="`maintain_order` has no effect and is only kept around for backwards-compatibility.",
    ):
        # this warns that maintain_order has no effect
        series.to_frame().select(nw_v1.col("a").unique(maintain_order=False).sum())


def test_head_aggregation() -> None:
    with pytest.raises(InvalidOperationError):
        nw_v1.col("a").mean().head()


def test_deprecated_expr_methods() -> None:
    data = {"a": [0, 0, 2, -1]}
    df = nw_v1.from_native(pd.DataFrame(data), eager_only=True)
    result = df.select(
        c=nw_v1.col("a").sort().head(2),
        d=nw_v1.col("a").sort().tail(2),
        e=(nw_v1.col("a") == 0).arg_true(),
        f=nw_v1.col("a").gather_every(2),
    )
    expected = {"c": [-1, 0], "d": [0, 2], "e": [0, 1], "f": [0, 2]}
    assert_equal_data(result, expected)

    with pytest.deprecated_call():
        df.select(
            c=nw.col("a").sort().head(2),
            d=nw.col("a").sort().tail(2),
            e=(nw.col("a") == 0).arg_true(),
            f=nw.col("a").gather_every(2),
        )


def test_dask_order_dependent_ops() -> None:
    # Preserve these for narwhals.stable.v1, even though they
    # raise after stable.v1.
    pytest.importorskip("dask")
    import dask.dataframe as dd

    df = nw_v1.from_native(dd.from_pandas(pd.DataFrame({"a": [1, 2, 3]})))
    result = df.select(
        a=nw_v1.col("a").cum_sum(),
        b=nw_v1.col("a").cum_count(),
        c=nw_v1.col("a").cum_prod(),
        d=nw_v1.col("a").cum_max(),
        e=nw_v1.col("a").cum_min(),
        f=nw_v1.col("a").shift(1),
        g=nw_v1.col("a").diff(),
        h=nw_v1.col("a").is_first_distinct(),
        i=nw_v1.col("a").is_last_distinct(),
    )
    expected = {
        "a": [1, 3, 6],
        "b": [1, 2, 3],
        "c": [1, 2, 6],
        "d": [1, 2, 3],
        "e": [1, 1, 1],
        "f": [None, 1.0, 2.0],
        "g": [None, 1.0, 1.0],
        "h": [True, True, True],
        "i": [True, True, True],
    }
    assert_equal_data(result, expected)
