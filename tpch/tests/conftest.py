from __future__ import annotations

from contextlib import suppress
from importlib.util import find_spec
from typing import TYPE_CHECKING

import polars as pl
import pytest

from tpch.classes import Backend, Query
from tpch.constants import DATA_DIR, METADATA_PATH

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from tpch.typing_ import QueryID


LINEITEM_PATH = DATA_DIR / "lineitem.parquet"
REGION_PATH = DATA_DIR / "region.parquet"
NATION_PATH = DATA_DIR / "nation.parquet"
SUPPLIER_PATH = DATA_DIR / "supplier.parquet"
PART_PATH = DATA_DIR / "part.parquet"
PARTSUPP_PATH = DATA_DIR / "partsupp.parquet"
ORDERS_PATH = DATA_DIR / "orders.parquet"
CUSTOMER_PATH = DATA_DIR / "customer.parquet"

SCALE_FACTORS_BLESSED = frozenset(
    (1.0, 10.0, 30.0, 100.0, 300.0, 1_000.0, 3_000.0, 10_000.0, 30_000.0, 100_000.0)
)
"""`scale_factor` values that are listed on [TPC-H v3.0.1 (Page 79)].

Using any other value *can* lead to incorrect results.

[TPC-H_v3.0.1 (Page 79)]: https://www.tpc.org/TPC_Documents_Current_Versions/pdf/TPC-H_v3.0.1.pdf
"""

SCALE_FACTORS_QUITE_SAFE = frozenset(
    (
        0.014,
        0.02,
        0.029,
        0.04,
        0.052,
        0.06,
        0.072,
        0.081,
        0.091,
        0.1,
        0.13,
        0.23,
        0.25,
        0.275,
        0.29,
        0.3,
        0.43,
        0.51,
    )
)
"""scale_factor` values that are **lower** than [TPC-H v3.0.1 (Page 79)], but still work fine.

[TPC-H_v3.0.1 (Page 79)]: https://www.tpc.org/TPC_Documents_Current_Versions/pdf/TPC-H_v3.0.1.pdf
"""


def pytest_addoption(parser: pytest.Parser) -> None:
    from tests.conftest import DEFAULT_CONSTRUCTORS

    parser.addoption(
        "--constructors",
        action="store",
        default=DEFAULT_CONSTRUCTORS,
        type=str,
        help="<sink for defaults in VSC getting injected>",
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


def q(query_id: QueryID, *paths: Path) -> Query:
    return Query(query_id, paths)


def iter_queries() -> Iterator[Query]:
    safe = SCALE_FACTORS_BLESSED | SCALE_FACTORS_QUITE_SAFE
    yield from (
        q("q1", LINEITEM_PATH),
        q("q2", REGION_PATH, NATION_PATH, SUPPLIER_PATH, PART_PATH, PARTSUPP_PATH),
        q("q3", CUSTOMER_PATH, LINEITEM_PATH, ORDERS_PATH),
        q("q4", LINEITEM_PATH, ORDERS_PATH),
        q(
            "q5",
            REGION_PATH,
            NATION_PATH,
            CUSTOMER_PATH,
            LINEITEM_PATH,
            ORDERS_PATH,
            SUPPLIER_PATH,
        ),
        q("q6", LINEITEM_PATH),
        q("q7", NATION_PATH, CUSTOMER_PATH, LINEITEM_PATH, ORDERS_PATH, SUPPLIER_PATH),
        q(
            "q8",
            PART_PATH,
            SUPPLIER_PATH,
            LINEITEM_PATH,
            ORDERS_PATH,
            CUSTOMER_PATH,
            NATION_PATH,
            REGION_PATH,
        ),
        q(
            "q9",
            PART_PATH,
            PARTSUPP_PATH,
            NATION_PATH,
            LINEITEM_PATH,
            ORDERS_PATH,
            SUPPLIER_PATH,
        ),
        q("q10", CUSTOMER_PATH, NATION_PATH, LINEITEM_PATH, ORDERS_PATH),
        q("q11", NATION_PATH, PARTSUPP_PATH, SUPPLIER_PATH),
        q("q12", LINEITEM_PATH, ORDERS_PATH),
        q("q13", CUSTOMER_PATH, ORDERS_PATH),
        q("q14", LINEITEM_PATH, PART_PATH),
        q("q15", LINEITEM_PATH, SUPPLIER_PATH),
        q("q16", PART_PATH, PARTSUPP_PATH, SUPPLIER_PATH),
        q("q17", LINEITEM_PATH, PART_PATH)
        .with_xfail(
            lambda _, scale_factor: (scale_factor < 0.014),
            reason="Generated dataset is too small, leading to 0 rows after the first two filters in `query1`.",
        )
        .with_skip(
            lambda _, scale_factor: scale_factor not in safe,
            reason="Non-determistic fails for `duckdb`, `sqlframe`. All other always fail, except `pyarrow` which always passes 🤯.",
        ),
        q("q18", CUSTOMER_PATH, LINEITEM_PATH, ORDERS_PATH),
        q("q19", LINEITEM_PATH, PART_PATH),
        q("q20", PART_PATH, PARTSUPP_PATH, NATION_PATH, LINEITEM_PATH, SUPPLIER_PATH),
        q("q21", LINEITEM_PATH, NATION_PATH, ORDERS_PATH, SUPPLIER_PATH).with_skip(
            lambda _, scale_factor: scale_factor not in safe, reason="Off-by-1 error"
        ),
        q("q22", CUSTOMER_PATH, ORDERS_PATH),
    )


@pytest.fixture(params=iter_queries(), ids=repr)
def query(request: pytest.FixtureRequest) -> Query:
    result: Query = request.param
    return result


@pytest.fixture(scope="session")
def scale_factor() -> float:
    df = pl.read_csv(METADATA_PATH, try_parse_dates=True)
    return float(df.get_column("scale_factor").item())
