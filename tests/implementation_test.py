from __future__ import annotations

# Using pyright's assert type instead
# mypy: disable-error-code="assert-type"
from typing import TYPE_CHECKING, Any, cast

import pytest

import narwhals as nw

if TYPE_CHECKING:
    from narwhals._namespace import _ModinDataFrame
    from narwhals._typing import (
        _ArrowImpl,
        _DaskImpl,
        _DuckDBImpl,
        _EagerAllowedImpl,
        _IbisImpl,
        _LazyAllowedImpl,
        _ModinImpl,
        _PandasImpl,
        _PolarsImpl,
        _SQLFrameImpl,
    )
    from narwhals.typing import IntoDataFrame, IntoLazyFrame, IntoSeries


def test_implementation_pandas() -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    assert (
        nw.from_native(pd.DataFrame({"a": [1, 2, 3]})).implementation
        is nw.Implementation.PANDAS
    )
    assert (
        nw.from_native(pd.DataFrame({"a": [1, 2, 3]}))["a"].implementation
        is nw.Implementation.PANDAS
    )
    assert nw.from_native(pd.DataFrame({"a": [1, 2, 3]})).implementation.is_pandas()
    assert nw.from_native(pd.DataFrame({"a": [1, 2, 3]})).implementation.is_pandas_like()


def test_implementation_polars() -> None:
    pytest.importorskip("polars")
    import polars as pl

    assert not nw.from_native(pl.DataFrame({"a": [1, 2, 3]})).implementation.is_pandas()
    assert not nw.from_native(pl.DataFrame({"a": [1, 2, 3]}))[
        "a"
    ].implementation.is_pandas()
    assert nw.from_native(pl.DataFrame({"a": [1, 2, 3]})).implementation.is_polars()
    assert nw.from_native(pl.LazyFrame({"a": [1, 2, 3]})).implementation.is_polars()


@pytest.mark.parametrize(
    ("member", "value"),
    [
        ("PANDAS", "pandas"),
        ("MODIN", "modin"),
        ("CUDF", "cudf"),
        ("PYARROW", "pyarrow"),
        ("PYSPARK", "pyspark"),
        ("POLARS", "polars"),
        ("DASK", "dask"),
        ("DUCKDB", "duckdb"),
        ("IBIS", "ibis"),
        ("SQLFRAME", "sqlframe"),
        ("PYSPARK_CONNECT", "pyspark[connect]"),
        ("UNKNOWN", "unknown"),
    ],
)
def test_implementation_new(member: str, value: str) -> None:
    assert nw.Implementation(value) is getattr(nw.Implementation, member)


if TYPE_CHECKING:
    import dask.dataframe as dd
    import duckdb
    import ibis
    import modin.pandas as mpd
    import pandas as pd
    import polars as pl
    import pyarrow as pa
    from sqlframe.base.dataframe import BaseDataFrame
    from typing_extensions import assert_type

    any_df: nw.DataFrame[Any] = cast("nw.DataFrame[Any]", "")
    any_ldf: nw.LazyFrame[Any] = cast("nw.LazyFrame[Any]", "")
    any_ser: nw.Series[Any] = cast("nw.Series[Any]", "")
    bound_df: nw.DataFrame[IntoDataFrame] = cast("nw.DataFrame[IntoDataFrame]", "")
    bound_ldf: nw.LazyFrame[IntoLazyFrame] = cast("nw.LazyFrame[IntoLazyFrame]", "")
    bound_ser: nw.Series[IntoSeries] = cast("nw.Series[IntoSeries]", "")

    def test_polars_typing(native: pl.DataFrame) -> None:
        df = nw.from_native(native)
        ldf = nw.from_native(native.lazy())
        ser = nw.from_native(native.to_series(), series_only=True)

        df_impl = df.implementation
        ldf_impl = ldf.implementation
        ser_impl = ser.implementation

        # [True Positive]
        any_df.lazy(df_impl)
        any_df.lazy(ldf_impl)
        any_df.lazy(ser_impl)
        any_ldf.collect(df_impl)
        any_ldf.collect(ldf_impl)
        any_ldf.collect(ser_impl)

        assert_type(df_impl, _PolarsImpl)
        assert_type(ldf_impl, _PolarsImpl)
        assert_type(ser_impl, _PolarsImpl)

    def test_pandas_typing(native: pd.DataFrame) -> None:
        df = nw.from_native(native)
        ldf = nw.from_native(native).lazy()
        ser = nw.from_native(native.iloc[0], series_only=True)

        df_impl = df.implementation
        ldf_impl = ldf.implementation
        ser_impl = ser.implementation

        # [True Negative]
        any_df.lazy(df_impl)  # type: ignore[arg-type]
        # [False Positive]
        any_df.lazy(ldf_impl)
        # [True Negative]
        any_df.lazy(ser_impl)  # pyright: ignore[reportArgumentType]
        # [True Positive]
        any_ldf.collect(df_impl)
        any_ldf.collect(ldf_impl)
        any_ldf.collect(ser_impl)

        assert_type(df_impl, _PandasImpl)
        # NOTE: Would require adding overloads to `DataFrame.lazy`
        assert_type(ldf_impl, _PandasImpl)  # pyright: ignore[reportAssertTypeFailure]
        assert_type(ser_impl, _PandasImpl)

    def test_arrow_typing(native: pa.Table) -> None:
        df = nw.from_native(native)
        ldf = nw.from_native(native).lazy()
        ser = nw.from_native(native.column(0), series_only=True)

        df_impl = df.implementation
        ldf_impl = ldf.implementation
        ser_impl = ser.implementation

        # [True Negative]
        any_df.lazy(df_impl)  # type: ignore[arg-type]
        # [False Positive]
        any_df.lazy(ldf_impl)
        # [True Negative]
        any_df.lazy(ser_impl)  # pyright: ignore[reportArgumentType]
        # [True Positive]
        any_ldf.collect(df_impl)
        any_ldf.collect(ldf_impl)
        any_ldf.collect(ser_impl)

        assert_type(df_impl, _ArrowImpl)
        # NOTE: Would require adding overloads to `DataFrame.lazy`
        assert_type(ldf_impl, _ArrowImpl)  # pyright: ignore[reportAssertTypeFailure]
        assert_type(ser_impl, _ArrowImpl)

    def test_duckdb_typing(native: duckdb.DuckDBPyRelation) -> None:
        ldf = nw.from_native(native)

        ldf_impl = ldf.implementation

        # [True Positive]
        any_df.lazy(ldf_impl)
        # [True Negative]
        any_ldf.collect(ldf_impl)  # type: ignore[arg-type]

        assert_type(ldf.implementation, _DuckDBImpl)

    def test_sqlframe_typing(native: BaseDataFrame[Any, Any, Any, Any, Any]) -> None:
        ldf = nw.from_native(native)

        ldf_impl = ldf.implementation

        # [True Negative]
        any_df.lazy(ldf_impl)  # pyright: ignore[reportArgumentType]
        any_ldf.collect(ldf_impl)  # pyright: ignore[reportArgumentType]

        assert_type(ldf.implementation, _SQLFrameImpl)

    def test_ibis_typing(native: ibis.Table) -> None:
        ldf = nw.from_native(native)

        ldf_impl = ldf.implementation

        # [True Positive]
        any_df.lazy(ldf_impl)
        # [True Negative]
        any_ldf.collect(ldf_impl)  # pyright: ignore[reportArgumentType]

        assert_type(ldf.implementation, _IbisImpl)

    def test_dask_typing(native: dd.DataFrame) -> None:
        ldf = nw.from_native(native)

        ldf_impl = ldf.implementation

        # [True Positive]
        any_df.lazy(ldf_impl)
        # [True Negative]
        any_ldf.collect(ldf_impl)  # pyright: ignore[reportArgumentType]

        assert_type(ldf.implementation, _DaskImpl)

    def test_modin_typing(native: mpd.DataFrame) -> None:
        df = nw.from_native(native)

        df_impl = df.implementation

        # [True Negative]
        any_df.lazy(df_impl)  # pyright: ignore[reportArgumentType]
        any_ldf.collect(df_impl)  # pyright: ignore[reportArgumentType]

        assert_type(df.implementation, _ModinImpl)  # pyright: ignore[reportAssertTypeFailure]
        # Fallback, remove if the above starts passing
        assert_type(df.implementation, _EagerAllowedImpl)

        # TODO @dangotbanned: Fix incompatible `_BasePandasLike.rename` signature
        # When this ignore isn't needed - the overload to `_ModinImpl` will work
        oops: _ModinDataFrame = native  # pyright: ignore[reportAssignmentType]  # noqa: F841

    def test_any_typing() -> None:
        df_impl = any_df.implementation
        ldf_impl = any_ldf.implementation
        ser_impl = any_ser.implementation

        # [False Positive]
        any_df.lazy(df_impl)
        any_df.lazy(ldf_impl)
        any_df.lazy(ser_impl)
        any_ldf.collect(df_impl)
        any_ldf.collect(ldf_impl)
        any_ldf.collect(ser_impl)

        assert_type(df_impl, _EagerAllowedImpl)  # pyright: ignore[reportAssertTypeFailure]
        assert_type(ldf_impl, _LazyAllowedImpl)  # pyright: ignore[reportAssertTypeFailure]
        assert_type(ser_impl, _EagerAllowedImpl)  # pyright: ignore[reportAssertTypeFailure]
        # Fallback, matches the first overload `_PolarsImpl`
        assert_type(df_impl, _PolarsImpl)
        assert_type(ldf_impl, _PolarsImpl)
        assert_type(ser_impl, _PolarsImpl)

    def test_bound_typing() -> None:
        df_impl = bound_df.implementation
        ldf_impl = bound_ldf.implementation
        ser_impl = bound_ser.implementation

        # [True Negative]
        any_df.lazy(df_impl)  # type: ignore[arg-type]
        any_df.lazy(ldf_impl)  # type: ignore[arg-type]
        any_df.lazy(ser_impl)  # type: ignore[arg-type]
        any_ldf.collect(df_impl)  # type: ignore[arg-type]
        any_ldf.collect(ldf_impl)  # type: ignore[arg-type]
        any_ldf.collect(ser_impl)  # type: ignore[arg-type]

        assert_type(df_impl, _EagerAllowedImpl)
        assert_type(ldf_impl, _LazyAllowedImpl)
        assert_type(ser_impl, _EagerAllowedImpl)

    def test_mixed_eager_typing(
        *args: nw.DataFrame[pl.DataFrame | pd.DataFrame | pa.Table]
        | nw.Series[pl.Series | pd.Series[Any] | pa.ChunkedArray[Any]],
    ) -> None:
        # NOTE: Any combination of eager objects that **does not** include `cuDF`, `modin` should
        # preserve that detail
        mix_impl = args[0].implementation

        # [True Negative]
        any_df.lazy(mix_impl)  # type: ignore[arg-type]
        # [True Positive]
        any_ldf.collect(mix_impl)

        assert_type(mix_impl, _PolarsImpl | _PandasImpl | _ArrowImpl)
