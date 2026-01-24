from __future__ import annotations

from contextlib import suppress
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl
import pytest

import narwhals as nw

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from narwhals._typing import BackendName
    from tpch.typing_ import DataLoader, TPCHBackend


# Data paths relative to tpch directory
TPCH_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = TPCH_DIR / "data"

LINEITEM_PATH = DATA_DIR / "lineitem.parquet"
REGION_PATH = DATA_DIR / "region.parquet"
NATION_PATH = DATA_DIR / "nation.parquet"
SUPPLIER_PATH = DATA_DIR / "supplier.parquet"
PART_PATH = DATA_DIR / "part.parquet"
PARTSUPP_PATH = DATA_DIR / "partsupp.parquet"
ORDERS_PATH = DATA_DIR / "orders.parquet"
CUSTOMER_PATH = DATA_DIR / "customer.parquet"

TPCH_TO_BACKEND_NAME: Mapping[TPCHBackend, BackendName] = {
    "polars[lazy]": "polars",
    "pyarrow": "pyarrow",
    "pandas[pyarrow]": "pandas",
    "dask": "dask",
    "duckdb": "duckdb",
    "sqlframe": "sqlframe",
}
def _build_backend_kwargs_map() -> dict[TPCHBackend, dict[str, Any]]:
    backend_map: dict[TPCHBackend, dict[str, Any]] = {"polars[lazy]": {}}

    pyarrow_installed = find_spec("pyarrow")

    if pyarrow_installed:
        backend_map["pyarrow"] = {}

    if pyarrow_installed and find_spec("pandas"):
        import pandas as pd

        # These options are deprecated in pandas >= 3.0 but needed for older versions
        with suppress(Exception):
            pd.options.mode.copy_on_write = True

        with suppress(Exception):
            pd.options.future.infer_string = True  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]

        backend_map["pandas[pyarrow]"] = {"engine": "pyarrow", "dtype_backend": "pyarrow"}

    if pyarrow_installed and find_spec("dask") and find_spec("dask.dataframe"):
        backend_map["dask"] = {"engine": "pyarrow", "dtype_backend": "pyarrow"}

    if find_spec("duckdb"):
        backend_map["duckdb"] = {}

    if find_spec("sqlframe"):
        from sqlframe.duckdb import DuckDBSession

        backend_map["sqlframe"] = {"session": DuckDBSession()}

    return backend_map


BACKEND_KWARGS_MAP = _build_backend_kwargs_map()

# Queries that need to be skipped for certain backends
DUCKDB_SKIPS = frozenset(
    [
        "q15"  # needs `filter` which works with window expressions
    ]
)


def _get_skip_backends(query_id: str) -> frozenset[str]:
    """Return backends that should be skipped for a given query."""
    if query_id in DUCKDB_SKIPS:
        return frozenset({"duckdb", "sqlframe"})
    return frozenset()


def skip_if_unsupported(query_id: str, backend_name: TPCHBackend) -> None:
    """Skip the test if the query is not supported for the given backend."""
    skip_backends = _get_skip_backends(query_id)
    if backend_name in skip_backends:
        pytest.skip(f"Query {query_id} is not supported for {backend_name}")


# Mapping of query IDs to their required data paths
QUERY_DATA_PATH_MAP: dict[str, tuple[Path, ...]] = {
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


@pytest.fixture(params=list(QUERY_DATA_PATH_MAP.keys()))
def query_id(request: pytest.FixtureRequest) -> str:
    """Fixture that yields each query_id."""
    return request.param  # type: ignore[no-any-return]


@pytest.fixture(params=list(BACKEND_KWARGS_MAP.keys()))
def backend_name(request: pytest.FixtureRequest) -> TPCHBackend:
    """Fixture that yields each backend name."""
    result: TPCHBackend = request.param
    return result


@pytest.fixture
def data_loader(backend_name: TPCHBackend) -> DataLoader:
    """Fixture that returns a function to load data for a given query.

    The returned function takes a query_id and returns a tuple of DataFrames
    in the order expected by that query's function signature.
    """
    kwargs = BACKEND_KWARGS_MAP[backend_name]
    backend = TPCH_TO_BACKEND_NAME[backend_name]

    def _load_data(query_id: str) -> tuple[nw.LazyFrame[Any], ...]:
        data_paths = QUERY_DATA_PATH_MAP[query_id]
        return tuple(
            nw.scan_parquet(str(path), backend=backend, **kwargs) for path in data_paths
        )

    return _load_data


@pytest.fixture
def expected_result() -> Callable[[str], pl.DataFrame]:
    """Fixture that returns a function to load expected results for a query."""

    def _load_expected(query_id: str) -> pl.DataFrame:
        return pl.read_parquet(DATA_DIR / f"result_{query_id}.parquet")

    return _load_expected
