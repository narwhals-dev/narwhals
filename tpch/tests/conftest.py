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
    from tpch.typing_ import DataLoader, QueryID, TPCHBackend


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


def pytest_addoption(parser: pytest.Parser) -> None:
    from tests.conftest import DEFAULT_CONSTRUCTORS

    parser.addoption(
        "--constructors",
        action="store",
        default=DEFAULT_CONSTRUCTORS,
        type=str,
        help="<sink for defaults in VSC getting injected>",
    )


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


QUERY_DATA_PATH_MAP: dict[QueryID, tuple[Path, ...]] = {
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
def query_id(request: pytest.FixtureRequest) -> QueryID:
    result: QueryID = request.param
    return result


@pytest.fixture(params=list(BACKEND_KWARGS_MAP.keys()))
def backend_name(request: pytest.FixtureRequest) -> TPCHBackend:
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

    def _load_data(query_id: QueryID) -> tuple[nw.LazyFrame[Any], ...]:
        data_paths = QUERY_DATA_PATH_MAP[query_id]
        return tuple(
            nw.scan_parquet(path.as_posix(), backend=backend, **kwargs)
            for path in data_paths
        )

    return _load_data


@pytest.fixture
def expected_result() -> Callable[[QueryID], pl.DataFrame]:
    """Fixture that returns a function to load expected results for a query."""

    def _load_expected(query_id: QueryID) -> pl.DataFrame:
        return pl.read_parquet(DATA_DIR / f"result_{query_id}.parquet")

    return _load_expected
