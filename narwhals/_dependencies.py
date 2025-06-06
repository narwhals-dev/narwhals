from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING

from narwhals.exceptions import module_not_found

if TYPE_CHECKING:
    from typing_extensions import Never


def import_polars(**_: Never):
    if find_spec("polars"):
        import polars as pl

        return pl

    raise module_not_found(module_name="polars")


def import_pandas(**_: Never):
    if find_spec("pandas"):
        import pandas as pd

        return pd

    raise module_not_found(module_name="pandas")


def import_modin(**_: Never):
    if find_spec("modin"):
        import modin.pandas

        return modin.pandas

    raise module_not_found(module_name="modin.pandas")


def import_cudf(**_: Never):  # pragma: no cover
    if find_spec("cudf"):
        import cudf

        return cudf

    raise module_not_found(module_name="cudf")


def import_pyarrow(**_: Never):
    if find_spec("pyarrow"):
        import pyarrow as pa

        return pa

    raise module_not_found(module_name="pyarrow")


def import_pyspark(**_: Never):  # pragma: no cover
    if find_spec("pyspark") and find_spec("pyspark.sql"):
        import pyspark.sql

        return pyspark.sql

    raise module_not_found(module_name="pyspark.sql")


def import_dask(**_: Never):
    if find_spec("dask"):
        import dask.dataframe

        return dask.dataframe

    raise module_not_found(module_name="dask.dataframe")


def import_duckdb(**_: Never):
    if find_spec("duckdb"):
        import duckdb

        return duckdb

    raise module_not_found(module_name="duckdb")


def import_sqlframe(**_: Never):
    if find_spec("sqlframe"):
        import sqlframe

        return sqlframe

    raise module_not_found(module_name="sqlframe")


def import_ibis(**_: Never):
    if find_spec("ibis"):
        import ibis

        return ibis

    raise module_not_found(module_name="ibis")


def import_pyspark_connect(**_: Never):  # pragma: no cover
    if (
        find_spec("pyspark")
        and find_spec("pyspark.sql")
        and find_spec("pyspark.sql.connect")
    ):
        import pyspark.sql.connect

        return pyspark.sql.connect

    raise module_not_found(module_name="pyspark.sql.connect")
