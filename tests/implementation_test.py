from __future__ import annotations

# Using pyright's assert type instead
# mypy: disable-error-code="assert-type"
from typing import TYPE_CHECKING, Any, cast

import pytest

import narwhals as nw

if TYPE_CHECKING:
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

    def test_implementation_typing() -> None:  # noqa: PLR0914, PLR0915
        import dask.dataframe as dd
        import modin.pandas as mpd
        import pandas as pd
        import polars as pl
        import pyarrow as pa
        from typing_extensions import assert_type

        from tests.conftest import (
            duckdb_lazy_constructor,
            ibis_lazy_constructor,
            sqlframe_pyspark_lazy_constructor,
        )

        data: dict[str, Any] = {"a": [1, 2, 3]}
        polars_df = nw.from_native(pl.DataFrame(data))
        polars_ldf = nw.from_native(pl.LazyFrame(data))
        polars_ser = nw.from_native(pl.Series(data["a"]), series_only=True)
        pandas_df = nw.from_native(pd.DataFrame(data))
        pandas_ser = nw.from_native(pd.Series(data["a"]), series_only=True)
        arrow_df = nw.from_native(pa.table(data))
        # NOTE: The overloads are too complicated, simplifying to `Any`
        arrow_ser_native = cast("pa.ChunkedArray[Any]", pa.chunked_array([data["a"]]))  # type: ignore[redundant-cast]
        arrow_ser = nw.from_native(arrow_ser_native, series_only=True)
        duckdb_ldf = nw.from_native(duckdb_lazy_constructor(data))
        sqlframe_ldf = nw.from_native(sqlframe_pyspark_lazy_constructor(data))
        ibis_ldf = nw.from_native(ibis_lazy_constructor(data))
        any_df = cast("nw.DataFrame[Any]", "fake df 1")
        any_ldf = cast("nw.LazyFrame[Any]", "fake ldf 1")
        any_ser = cast("nw.Series[Any]", "fake ser 1")
        bound_df = cast("nw.DataFrame[IntoDataFrame]", "fake df 2")
        bound_ldf = cast("nw.LazyFrame[IntoLazyFrame]", "fake ldf 2")
        bound_ser = cast("nw.Series[IntoSeries]", "fake ser 2")

        polars_df_impl = polars_df.implementation
        polars_ldf_impl = polars_ldf.implementation
        polars_ser_impl = polars_ser.implementation
        pandas_df_impl = pandas_df.implementation
        pandas_ser_impl = pandas_ser.implementation
        arrow_df_impl = arrow_df.implementation
        arrow_ser_impl = arrow_ser.implementation
        duckdb_impl = duckdb_ldf.implementation
        sqlframe_impl = sqlframe_ldf.implementation
        ibis_impl = ibis_ldf.implementation

        assert_type(polars_df_impl, _PolarsImpl)
        assert_type(polars_ldf_impl, _PolarsImpl)
        assert_type(polars_ser_impl, _PolarsImpl)
        # NOTE: Testing the lazy versions of pandas/pyarrow would require adding overloads to `DataFrame.lazy`
        # Currently, everything becomes `LazyFrame[Any]`
        assert_type(pandas_df_impl, _PandasImpl)
        assert_type(pandas_ser_impl, _PandasImpl)
        assert_type(arrow_df_impl, _ArrowImpl)
        assert_type(arrow_ser_impl, _ArrowImpl)
        assert_type(duckdb_impl, _DuckDBImpl)
        assert_type(sqlframe_impl, _SQLFrameImpl)

        modin_native = mpd.DataFrame.from_dict(data)
        modin_df = nw.from_native(modin_native)
        modin_impl = modin_df.implementation
        # TODO @dangotbanned: Is this even possible?
        # - `mypy` won't ever work, treats as `Any`
        # - `pyright` can resolve `modin_df: narwhals.dataframe.DataFrame[modin.pandas.dataframe.DataFrame]`
        #   - But we run into variance issues if trying to widen the concrete type again
        assert_type(modin_impl, _ModinImpl)  # pyright: ignore[reportAssertTypeFailure]
        # If ^^^ can be fixed, the next one should be removed
        assert_type(modin_impl, _EagerAllowedImpl)

        # NOTE: Constructor returns `Unknown`
        dask_native = cast("dd.DataFrame", dd.DataFrame.from_dict(data))
        dask_ldf = nw.from_native(dask_native)
        dask_impl = dask_ldf.implementation
        # NOTE: Same issue as modin
        assert_type(dask_impl, _DaskImpl)  # pyright: ignore[reportAssertTypeFailure]
        # If ^^^ can be fixed, the next one should be removed
        assert_type(dask_impl, _LazyAllowedImpl)

        # NOTE: Also same issue 🤔
        # TODO @dangotbanned: try something else instead
        assert_type(ibis_impl, _IbisImpl)  # pyright: ignore[reportAssertTypeFailure]
        assert_type(dask_impl, _LazyAllowedImpl)

        # NOTE: Any combination of eager objects that **does not** include `cuDF`, `modin` should
        # preserve that detail
        can_lazyframe_collect_dfs: list[
            nw.DataFrame[pl.DataFrame]
            | nw.DataFrame[pd.DataFrame]
            | nw.DataFrame[pa.Table]
        ] = [polars_df, pandas_df, arrow_df]
        can_lazyframe_collect_dfs_impl = can_lazyframe_collect_dfs[0].implementation
        assert_type(
            can_lazyframe_collect_dfs_impl, _PolarsImpl | _PandasImpl | _ArrowImpl
        )
        can_lazyframe_collect_sers: list[
            nw.Series[pl.Series]
            | nw.Series[pd.Series[Any]]
            | nw.Series[pa.ChunkedArray[Any]]
        ] = [polars_ser, pandas_ser, arrow_ser]
        can_lazyframe_collect_sers_impl = can_lazyframe_collect_sers[0].implementation
        assert_type(
            can_lazyframe_collect_sers_impl, _PolarsImpl | _PandasImpl | _ArrowImpl
        )

        any_df_impl = any_df.implementation
        any_ldf_impl = any_ldf.implementation
        any_ser_impl = any_ser.implementation
        # TODO @dangotbanned: Is this so bad?
        # - Currently `DataFrame[Any] | LazyFrame[Any] | Series[Any]` matches the first overload (`_PolarsImpl`)
        # - That is accepted **everywhere** that uses `IntoBackend`
        assert_type(any_df_impl, _EagerAllowedImpl)  # pyright: ignore[reportAssertTypeFailure]
        assert_type(any_ldf_impl, _LazyAllowedImpl)  # pyright: ignore[reportAssertTypeFailure]
        assert_type(any_ser_impl, _EagerAllowedImpl)  # pyright: ignore[reportAssertTypeFailure]

        bound_df_impl = bound_df.implementation
        bound_ldf_impl = bound_ldf.implementation
        bound_ser_impl = bound_ser.implementation
        assert_type(bound_df_impl, _EagerAllowedImpl)
        assert_type(bound_ldf_impl, _LazyAllowedImpl)
        assert_type(bound_ser_impl, _EagerAllowedImpl)

        # NOTE: `DataFrame.lazy`
        # [True Positive]
        any_df.lazy(polars_ldf.implementation)
        any_df.lazy(polars_df.implementation)
        any_df.lazy(duckdb_ldf.implementation)

        # [True Negative]
        any_df.lazy(pandas_df.implementation)  # type: ignore[arg-type]
        any_df.lazy(arrow_df.implementation)  # type: ignore[arg-type]
        any_df.lazy(modin_df.implementation)  # pyright: ignore[reportArgumentType]
        any_df.lazy(sqlframe_ldf.implementation)  # type: ignore[arg-type]
        any_df.lazy(bound_ldf.implementation)  # type: ignore[arg-type]
        any_df.lazy(bound_df.implementation)  # type: ignore[arg-type]
        any_df.lazy(bound_ser.implementation)  # type: ignore[arg-type]
        any_df.lazy(can_lazyframe_collect_dfs[0].implementation)  # type: ignore[arg-type]

        # [False Positive]
        any_df.lazy(any_ldf.implementation)
        any_df.lazy(any_df.implementation)
        any_df.lazy(any_ser.implementation)

        # [False Negative]
        any_df.lazy(ibis_ldf.implementation)  # pyright: ignore[reportArgumentType]
        any_df.lazy(dask_ldf.implementation)  # pyright: ignore[reportArgumentType]
