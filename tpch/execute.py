from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path

import dask.dataframe as dd
import duckdb
import pandas as pd
import polars as pl
import pyarrow as pa
from polars.testing import assert_frame_equal

import narwhals as nw

pd.options.mode.copy_on_write = True
pd.options.future.infer_string = True
pl.Config.set_fmt_float("full")

DATA_DIR = Path("data")
LINEITEM_PATH = DATA_DIR / "lineitem.parquet"
REGION_PATH = DATA_DIR / "region.parquet"
NATION_PATH = DATA_DIR / "nation.parquet"
SUPPLIER_PATH = DATA_DIR / "supplier.parquet"
PART_PATH = DATA_DIR / "part.parquet"
PARTSUPP_PATH = DATA_DIR / "partsupp.parquet"
ORDERS_PATH = DATA_DIR / "orders.parquet"
CUSTOMER_PATH = DATA_DIR / "customer.parquet"

BACKEND_NAMESPACE_KWARGS_MAP = {
    "pandas[pyarrow]": (pd, {"engine": "pyarrow", "dtype_backend": "pyarrow"}),
    "polars[lazy]": (pl, {}),
    "pyarrow": (pa, {}),
    "duckdb": (duckdb, {}),
    "dask": (dd, {"engine": "pyarrow", "dtype_backend": "pyarrow"}),
}

DUCKDB_SKIPS = ["q15"]

QUERY_DATA_PATH_MAP = {
    "q1": (LINEITEM_PATH,),
    "q2": (REGION_PATH, NATION_PATH, SUPPLIER_PATH, PART_PATH, PARTSUPP_PATH),
    "q3": (CUSTOMER_PATH, LINEITEM_PATH, ORDERS_PATH),
    "q4": (LINEITEM_PATH, ORDERS_PATH),
    "q5": (
        REGION_PATH,
        NATION_PATH,
        CUSTOMER_PATH,
        LINEITEM_PATH,
        ORDERS_PATH,
        SUPPLIER_PATH,
    ),
    "q6": (LINEITEM_PATH,),
    "q7": (NATION_PATH, CUSTOMER_PATH, LINEITEM_PATH, ORDERS_PATH, SUPPLIER_PATH),
    "q8": (
        PART_PATH,
        SUPPLIER_PATH,
        LINEITEM_PATH,
        ORDERS_PATH,
        CUSTOMER_PATH,
        NATION_PATH,
        REGION_PATH,
    ),
    "q9": (
        PART_PATH,
        PARTSUPP_PATH,
        NATION_PATH,
        LINEITEM_PATH,
        ORDERS_PATH,
        SUPPLIER_PATH,
    ),
    "q10": (CUSTOMER_PATH, NATION_PATH, LINEITEM_PATH, ORDERS_PATH),
    "q11": (NATION_PATH, PARTSUPP_PATH, SUPPLIER_PATH),
    "q12": (LINEITEM_PATH, ORDERS_PATH),
    "q13": (CUSTOMER_PATH, ORDERS_PATH),
    "q14": (LINEITEM_PATH, PART_PATH),
    "q15": (LINEITEM_PATH, SUPPLIER_PATH),
    "q16": (PART_PATH, PARTSUPP_PATH, SUPPLIER_PATH),
    "q17": (LINEITEM_PATH, PART_PATH),
    "q18": (CUSTOMER_PATH, LINEITEM_PATH, ORDERS_PATH),
    "q19": (LINEITEM_PATH, PART_PATH),
    "q20": (PART_PATH, PARTSUPP_PATH, NATION_PATH, LINEITEM_PATH, SUPPLIER_PATH),
    "q21": (LINEITEM_PATH, NATION_PATH, ORDERS_PATH, SUPPLIER_PATH),
    "q22": (CUSTOMER_PATH, ORDERS_PATH),
}


def execute_query(query_id: str) -> None:
    query_module = import_module(f"tpch.queries.{query_id}")
    data_paths = QUERY_DATA_PATH_MAP[query_id]

    expected = pl.read_parquet(DATA_DIR / f"result_{query_id}.parquet")

    for backend, (native_namespace, kwargs) in BACKEND_NAMESPACE_KWARGS_MAP.items():
        if backend == "duckdb" and query_id in DUCKDB_SKIPS:
            print(f"\nSkipping {query_id} for DuckDB")  # noqa: T201
            continue

        print(f"\nRunning {query_id} with {backend=}")  # noqa: T201
        result: pl.DataFrame = (
            query_module.query(
                *(
                    nw.scan_parquet(
                        str(path), native_namespace=native_namespace, **kwargs
                    )
                    for path in data_paths
                )
            )
            .collect(backend=nw.Implementation.POLARS)
            .to_native()
        )

        assert_frame_equal(expected, result, check_dtypes=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute a TPCH query by number.")
    parser.add_argument("query", type=str, help="The query to execute, e.g. 'q1'.")
    args = parser.parse_args()

    execute_query(query_id=args.query)


if __name__ == "__main__":
    main()
