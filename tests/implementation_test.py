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
        _EagerAllowedImpl,
        _LazyAllowedImpl,
        _ModinImpl,
        _PandasImpl,
        _PolarsImpl,
    )
    from narwhals.typing import IntoDataFrame


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

    def test_implementation_typing() -> None:  # noqa: PLR0914
        import dask.dataframe as dd
        import modin.pandas as mpd
        import pandas as pd
        import polars as pl
        import pyarrow as pa
        from typing_extensions import assert_type

        data: dict[str, Any] = {"a": [1, 2, 3]}
        polars_df = nw.from_native(pl.DataFrame(data))
        polars_ldf = nw.from_native(pl.LazyFrame(data))
        pandas_df = nw.from_native(pd.DataFrame(data))
        arrow_df = nw.from_native(pa.table(data))

        polars_impl = polars_df.implementation
        lazy_polars_impl = polars_ldf.implementation
        pandas_impl = pandas_df.implementation
        arrow_impl = arrow_df.implementation

        assert_type(polars_impl, _PolarsImpl)
        assert_type(lazy_polars_impl, _PolarsImpl)
        # NOTE: Testing the lazy versions of pandas/pyarrow would require adding overloads to `DataFrame.lazy`
        # Currently, everything becomes `LazyFrame[Any]`
        assert_type(pandas_impl, _PandasImpl)
        assert_type(arrow_impl, _ArrowImpl)

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

        can_lazyframe_collect_dfs: list[
            nw.DataFrame[pl.DataFrame]
            | nw.DataFrame[pd.DataFrame]
            | nw.DataFrame[pa.Table]
        ] = [polars_df, pandas_df, arrow_df]
        can_lazyframe_collect_impl = can_lazyframe_collect_dfs[0].implementation
        assert_type(can_lazyframe_collect_impl, _PolarsImpl | _PandasImpl | _ArrowImpl)

        very_lost_df = nw.DataFrame.__new__(nw.DataFrame)
        very_lost_impl = very_lost_df.implementation
        # TODO @dangotbanned: Is this so bad?
        # - Currently `DataFrame[Any]` matches the first overload (`_PolarsImpl`)
        # - That is accepted **everywhere** that uses `IntoBackend`
        assert_type(very_lost_impl, _EagerAllowedImpl)  # pyright: ignore[reportAssertTypeFailure]

        not_so_lost_df = nw.DataFrame.__new__(nw.DataFrame[IntoDataFrame])
        not_so_lost_impl = not_so_lost_df.implementation
        assert_type(not_so_lost_impl, _EagerAllowedImpl)
