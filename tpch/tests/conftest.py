from __future__ import annotations

from contextlib import suppress
from importlib.util import find_spec
from typing import TYPE_CHECKING

import pytest

from tpch.classes import Backend, Query
from tpch.constants import get_scale_factor_dir

if TYPE_CHECKING:
    from collections.abc import Iterator

    from tpch.typing_ import QueryID

# Table names used to construct paths dynamically
TBL_LINEITEM = "lineitem"
TBL_REGION = "region"
TBL_NATION = "nation"
TBL_SUPPLIER = "supplier"
TBL_PART = "part"
TBL_PARTSUPP = "partsupp"
TBL_ORDERS = "orders"
TBL_CUSTOMER = "customer"

SCALE_FACTORS_BLESSED = frozenset(
    (1.0, 10.0, 30.0, 100.0, 300.0, 1_000.0, 3_000.0, 10_000.0, 30_000.0, 100_000.0)
)
"""`scale_factor` values that are listed on [TPC-H v3.0.1 (Page 79)].

Using any other value *can* lead to incorrect results.

[TPC-H_v3.0.1 (Page 79)]: https://www.tpc.org/TPC_Documents_Current_Versions/pdf/TPC-H_v3.0.1.pdf
"""

SCALE_FACTORS_QUITE_SAFE = frozenset((0.1, 0.13, 0.23, 0.25, 0.275, 0.29, 0.3))
"""scale_factor` values that are **lower** than [TPC-H v3.0.1 (Page 79)], but still work fine.

[TPC-H_v3.0.1 (Page 79)]: https://www.tpc.org/TPC_Documents_Current_Versions/pdf/TPC-H_v3.0.1.pdf
"""

DEFAULT_SCALE_FACTOR = 0.1


def _scale_factor_data_exists(scale_factor: float) -> bool:
    """Check if data for the given scale factor exists by checking if its directory exists."""
    sf_dir = get_scale_factor_dir(scale_factor)
    return sf_dir.exists()


def pytest_configure(config: pytest.Config) -> None:
    """Generate TPC-H data if it doesn't exist for the requested scale factor.

    This hook runs after command line options have been parsed,
    ensuring data is available before test collection.
    """
    scale_factor = config.getoption("--scale-factor", default=DEFAULT_SCALE_FACTOR)

    if _scale_factor_data_exists(scale_factor):
        return

    # Import here to avoid circular imports and keep startup fast when data exists
    from tpch.generate_data import main as generate_data

    generate_data(scale_factor=scale_factor)


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
        default=DEFAULT_SCALE_FACTOR,
        type=float,
        help=f"TPC-H scale factor to use for tests (default: {DEFAULT_SCALE_FACTOR})",
    )


def iter_backends() -> Iterator[Backend]:
    yield Backend("polars[lazy]", "polars")
    if find_spec("pyarrow"):
        yield Backend("pyarrow", "pyarrow")
        if find_spec("pandas"):
            import pandas as pd

            # These options are deprecated in pandas >= 3.0 but needed for older versions
            with suppress(Exception):
                pd.options.mode.copy_on_write = True
            with suppress(Exception):
                pd.options.future.infer_string = True  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]
            yield Backend(
                "pandas[pyarrow]", "pandas", engine="pyarrow", dtype_backend="pyarrow"
            )
        if find_spec("dask") and find_spec("dask.dataframe"):
            yield Backend("dask", "dask", engine="pyarrow", dtype_backend="pyarrow")
    if find_spec("duckdb"):
        yield Backend("duckdb", "duckdb")
        if find_spec("sqlframe"):
            from sqlframe.duckdb import DuckDBSession

            yield Backend("sqlframe", "sqlframe", session=DuckDBSession())


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
        q("q15", TBL_LINEITEM, TBL_SUPPLIER).with_skip(
            lambda backend, _: backend.name in {"duckdb", "sqlframe"},
            reason="https://github.com/narwhals-dev/narwhals/issues/2226",
        ),
        q("q16", TBL_PART, TBL_PARTSUPP, TBL_SUPPLIER),
        q("q17", TBL_LINEITEM, TBL_PART).with_xfail(
            lambda _, scale_factor: (scale_factor < 0.014) or scale_factor == 0.5,
            reason="Generated dataset is too small, leading to 0 rows after the first two filters in `query1`.",
        ),
        q("q18", TBL_CUSTOMER, TBL_LINEITEM, TBL_ORDERS),
        q("q19", TBL_LINEITEM, TBL_PART),
        q("q20", TBL_PART, TBL_PARTSUPP, TBL_NATION, TBL_LINEITEM, TBL_SUPPLIER),
        q("q21", TBL_LINEITEM, TBL_NATION, TBL_ORDERS, TBL_SUPPLIER).with_skip(
            lambda _, scale_factor: scale_factor
            not in (SCALE_FACTORS_BLESSED | SCALE_FACTORS_QUITE_SAFE),
            reason="Off-by-1 error when using *most* non-blessed `scale_factor`s",
        ),
        q("q22", TBL_CUSTOMER, TBL_ORDERS),
    )


@pytest.fixture(params=iter_queries(), ids=repr)
def query(request: pytest.FixtureRequest) -> Query:
    result: Query = request.param
    return result


@pytest.fixture(scope="session")
def scale_factor(request: pytest.FixtureRequest) -> float:
    """Get the scale factor from pytest options."""
    return float(request.config.getoption("--scale-factor"))
