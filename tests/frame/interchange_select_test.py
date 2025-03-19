from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Mapping

import pytest

import narwhals.stable.v1 as nw

if TYPE_CHECKING:
    from typing_extensions import Self

data: Mapping[str, Any] = {"a": [1, 2, 3], "b": [4.0, 5.0, 6.1], "z": ["x", "y", "z"]}


class InterchangeDataFrame:
    def __init__(self: Self, df: CustomDataFrame) -> None:
        self._df = df

    def __dataframe__(self: Self) -> InterchangeDataFrame:  # pragma: no cover
        return self

    def column_names(self: Self) -> list[str]:
        return list(self._df._data.keys())

    def select_columns_by_name(self: Self, columns: list[str]) -> InterchangeDataFrame:
        return InterchangeDataFrame(
            CustomDataFrame(
                {key: value for key, value in self._df._data.items() if key in columns}
            )
        )


class CustomDataFrame:
    def __init__(self: Self, data: dict[str, Any]) -> None:
        self._data = data

    def __dataframe__(self: Self, *, allow_copy: bool = True) -> InterchangeDataFrame:
        return InterchangeDataFrame(self)


def test_interchange() -> None:
    df = CustomDataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "z": [1, 4, 2]})
    result = nw.from_native(df, eager_or_interchange_only=True).select("a", "z")
    assert result.columns == ["a", "z"]


def test_interchange_ibis(
    tmpdir: pytest.TempdirFactory, request: pytest.FixtureRequest
) -> None:  # pragma: no cover
    pytest.importorskip("polars")
    pytest.importorskip("ibis")

    import ibis
    import polars as pl

    try:
        ibis.set_backend("duckdb")
    except ImportError:
        request.applymarker(pytest.mark.xfail)
    df_pl = pl.DataFrame(data)

    filepath = str(tmpdir / "file.parquet")  # type: ignore[operator]
    df_pl.write_parquet(filepath)

    tbl = ibis.read_parquet(filepath)
    df = nw.from_native(tbl, eager_or_interchange_only=True)

    out_cols = df.select("a", "z").schema.names()

    assert out_cols == ["a", "z"]


def test_interchange_duckdb() -> None:
    pytest.importorskip("polars")
    pytest.importorskip("duckdb")

    import duckdb
    import polars as pl

    df_pl = pl.DataFrame(data)  # noqa: F841
    rel = duckdb.sql("select * from df_pl")
    df = nw.from_native(rel, eager_or_interchange_only=True)

    out_cols = df.select("a", "z").schema.names()

    assert out_cols == ["a", "z"]
