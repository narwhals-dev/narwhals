from __future__ import annotations

from typing import Literal

import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

import narwhals as nw
import narwhals.stable.v1 as nw_v1
from tests.utils import Constructor
from tests.utils import assert_equal_data


def test_collect_kwargs(constructor: Constructor) -> None:
    data = {"a": [1, 2], "b": [3, 4]}
    df = nw_v1.from_native(constructor(data))

    result = (
        df.lazy()
        .select(nw_v1.col("a", "b").sum())
        .collect(
            polars_kwargs={"no_optimization": True},
            dask_kwargs={"optimize_graph": False},
            duckdb_kwargs={"eager_backend": "pyarrow"},
        )
    )

    expected = {"a": [3], "b": [7]}
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("eager_backend", "expected_cls"),
    [
        ("pyarrow", pa.Table),
        ("polars", pl.DataFrame),
        ("pandas", pd.DataFrame),
    ],
)
def test_collect_duckdb(
    eager_backend: Literal["pyarrow", "polars", "pandas"], expected_cls: type
) -> None:
    duckdb = pytest.importorskip("duckdb")

    data = {"a": [1, 2], "b": [3, 4]}
    df_pl = pl.DataFrame(data)  # noqa: F841
    df = nw.from_native(duckdb.sql("select * from df_pl"))

    result = df.lazy().collect(duckdb_kwargs={"eager_backend": eager_backend}).to_native()
    assert isinstance(result, expected_cls)


def test_collect_duckdb_raise() -> None:
    duckdb = pytest.importorskip("duckdb")

    data = {"a": [1, 2], "b": [3, 4]}
    df_pl = pl.DataFrame(data)  # noqa: F841
    df = nw.from_native(duckdb.sql("select * from df_pl"))

    with pytest.raises(
        ValueError,
        match=(
            "Only the following `eager_backend`'s are supported: pyarrow, pandas and "
            "polars. Found 'foo'."
        ),
    ):
        df.lazy().collect(duckdb_kwargs={"eager_backend": "foo"})
