from __future__ import annotations

from contextlib import suppress
from importlib.util import find_spec
from typing import TYPE_CHECKING

import pytest

from tpch.classes import Backend, Query
from tpch.constants import SCALE_FACTOR_DEFAULT, SCALE_FACTORS

if TYPE_CHECKING:
    from collections.abc import Iterator

    from tpch.typing_ import QueryID, ScaleFactor

# Table names used to construct paths dynamically
TBL_LINEITEM = "lineitem"
TBL_REGION = "region"
TBL_NATION = "nation"
TBL_SUPPLIER = "supplier"
TBL_PART = "part"
TBL_PARTSUPP = "partsupp"
TBL_ORDERS = "orders"
TBL_CUSTOMER = "customer"


def is_xdist_worker(obj: pytest.FixtureRequest | pytest.Config, /) -> bool:
    # Adapted from https://github.com/pytest-dev/pytest-xdist/blob/8b60b1ef5d48974a1cb69bc1a9843564bdc06498/src/xdist/plugin.py#L337-L349
    return hasattr(obj if isinstance(obj, pytest.Config) else obj.config, "workerinput")


def pytest_configure(config: pytest.Config) -> None:
    """Generate TPC-H data if it doesn't exist for the requested scale factor.

    [`configure`] runs after `addoption`, ensuring data is available before test collection.

    [`configure`]: https://docs.pytest.org/en/stable/reference/reference.html#pytest.hookspec.pytest_configure
    """
    # Only run before the session starts, instead of 1 + (`--numprocesses`)
    if is_xdist_worker(config):
        return
    from tpch.generate_data import TPCHGen

    TPCHGen.from_pytest(config).run()


def pytest_addoption(parser: pytest.Parser) -> None:
    from tests.conftest import DEFAULT_CONSTRUCTORS

    parser.addoption(
        "--constructors",
        action="store",
        default=DEFAULT_CONSTRUCTORS,
        type=str,
        help="<sink for defaults in VSC getting injected>",
    )
    parser.addoption(
        "--scale-factor",
        action="store",
        default=SCALE_FACTOR_DEFAULT,
        choices=SCALE_FACTORS,
        help="TPC-H scale factor to use for tests (default: %(default)s)",
    )


def iter_backends() -> Iterator[Backend]:
    yield Backend("polars[lazy]")
    if find_spec("pyarrow"):
        yield Backend("pyarrow")
        if find_spec("pandas"):
            import pandas as pd

            # These options are deprecated in pandas >= 3.0 but needed for older versions
            with suppress(Exception):
                pd.options.mode.copy_on_write = True
            with suppress(Exception):
                pd.options.future.infer_string = True  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]
            yield Backend("pandas[pyarrow]", engine="pyarrow", dtype_backend="pyarrow")
        if find_spec("dask") and find_spec("dask.dataframe"):
            yield Backend("dask", engine="pyarrow", dtype_backend="pyarrow")
    if find_spec("duckdb"):
        yield Backend("duckdb")
        if find_spec("sqlframe"):
            from sqlframe.duckdb import DuckDBSession

            yield Backend("sqlframe", session=DuckDBSession())


@pytest.fixture(params=iter_backends(), ids=repr)
def backend(request: pytest.FixtureRequest) -> Backend:
    result: Backend = request.param
    return result


def q(query_id: QueryID, *table_names: str) -> Query:
    """Create a Query with table names (paths resolved at runtime based on scale_factor)."""
    return Query(query_id, table_names)


def iter_queries() -> Iterator[Query]:
    yield from (
        q("q1", TBL_LINEITEM),
        q("q2", TBL_REGION, TBL_NATION, TBL_SUPPLIER, TBL_PART, TBL_PARTSUPP),
        q("q3", TBL_CUSTOMER, TBL_LINEITEM, TBL_ORDERS),
        q("q4", TBL_LINEITEM, TBL_ORDERS),
        q(
            "q5",
            TBL_REGION,
            TBL_NATION,
            TBL_CUSTOMER,
            TBL_LINEITEM,
            TBL_ORDERS,
            TBL_SUPPLIER,
        ),
        q("q6", TBL_LINEITEM),
        q("q7", TBL_NATION, TBL_CUSTOMER, TBL_LINEITEM, TBL_ORDERS, TBL_SUPPLIER),
        q(
            "q8",
            TBL_PART,
            TBL_SUPPLIER,
            TBL_LINEITEM,
            TBL_ORDERS,
            TBL_CUSTOMER,
            TBL_NATION,
            TBL_REGION,
        ),
        q(
            "q9",
            TBL_PART,
            TBL_PARTSUPP,
            TBL_NATION,
            TBL_LINEITEM,
            TBL_ORDERS,
            TBL_SUPPLIER,
        ),
        q("q10", TBL_CUSTOMER, TBL_NATION, TBL_LINEITEM, TBL_ORDERS),
        q("q11", TBL_NATION, TBL_PARTSUPP, TBL_SUPPLIER),
        q("q12", TBL_LINEITEM, TBL_ORDERS),
        q("q13", TBL_CUSTOMER, TBL_ORDERS),
        q("q14", TBL_LINEITEM, TBL_PART),
        q("q15", TBL_LINEITEM, TBL_SUPPLIER),
        q("q16", TBL_PART, TBL_PARTSUPP, TBL_SUPPLIER),
        q("q17", TBL_LINEITEM, TBL_PART),
        q("q18", TBL_CUSTOMER, TBL_LINEITEM, TBL_ORDERS),
        q("q19", TBL_LINEITEM, TBL_PART),
        q("q20", TBL_PART, TBL_PARTSUPP, TBL_NATION, TBL_LINEITEM, TBL_SUPPLIER),
        q("q21", TBL_LINEITEM, TBL_NATION, TBL_ORDERS, TBL_SUPPLIER),
        q("q22", TBL_CUSTOMER, TBL_ORDERS),
    )


@pytest.fixture(scope="session")
def scale_factor(request: pytest.FixtureRequest) -> ScaleFactor:
    """Get the scale factor from pytest options."""
    return request.config.getoption("--scale-factor")


@pytest.fixture(params=iter_queries(), ids=repr)
def query(request: pytest.FixtureRequest, scale_factor: ScaleFactor) -> Query:
    result: Query = request.param
    return result.with_scale_factor(scale_factor)
