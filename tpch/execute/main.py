from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path

import pandas as pd
import polars as pl
import pyarrow.parquet as pq

pd.options.mode.copy_on_write = True
pd.options.future.infer_string = True

# Data paths
DATA_DIR = Path("data")
LINEITEM_PATH = DATA_DIR / "lineitem.parquet"
REGION_PATH = DATA_DIR / "region.parquet"
NATION_PATH = DATA_DIR / "nation.parquet"
SUPPLIER_PATH = DATA_DIR / "supplier.parquet"
PART_PATH = DATA_DIR / "part.parquet"
PARTSUPP_PATH = DATA_DIR / "partsupp.parquet"
ORDERS_PATH = DATA_DIR / "orders.parquet"
CUSTOMER_PATH = DATA_DIR / "customer.parquet"

# Read functions for each backend
READ_FUNCS = {
    # "pandas": lambda x: pd.read_parquet(x, engine="pyarrow"), # noqa: ERA001
    "pandas[pyarrow]": lambda x: pd.read_parquet(
        x, engine="pyarrow", dtype_backend="pyarrow"
    ),
    # "polars[eager]": lambda x: pl.read_parquet(x),
    "polars[lazy]": lambda x: pl.scan_parquet(x),
    "pyarrow": lambda x: pq.read_table(x),
    # "dask": lambda x: dd.read_parquet(x, engine="pyarrow", dtype_backend="pyarrow"), # noqa: ERA001
}

# Collect functions for the lazy backends
COLLECT_FUNCS = {
    "polars[lazy]": lambda x: x.collect(),
    # "dask": lambda x: x.compute(), # noqa: ERA001
}

QUERY_DATA = {
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
    data_paths = QUERY_DATA[query_id]

    for backend, read_func in READ_FUNCS.items():
        print(f"\nRunning {query_id} with {backend=}")
        result = query_module.query(*(read_func(path) for path in data_paths))
        if collect_func := COLLECT_FUNCS.get(backend):
            result = collect_func(result)
        print(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute a TPCH query by number.")
    parser.add_argument("query", type=str, help="The query to execute, e.g. 'q1'.")
    args = parser.parse_args()

    execute_query(query_id=args.query)


if __name__ == "__main__":
    main()
