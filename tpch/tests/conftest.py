from __future__ import annotations

from contextlib import suppress
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import polars as pl
import pytest

import narwhals as nw

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from narwhals._typing import IntoBackendAny
    from narwhals.typing import FileSource
    from tpch.typing_ import KnownImpl, QueryID, QueryModule, TPCHBackend


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


def pytest_addoption(parser: pytest.Parser) -> None:
    from tests.conftest import DEFAULT_CONSTRUCTORS

    parser.addoption(
        "--constructors",
        action="store",
        default=DEFAULT_CONSTRUCTORS,
        type=str,
        help="<sink for defaults in VSC getting injected>",
    )


class Backend:
    name: TPCHBackend
    implementation: KnownImpl
    skips: frozenset[QueryID]
    kwds: dict[str, Any]

    def __init__(
        self,
        name: TPCHBackend,
        into_backend: IntoBackendAny,
        /,
        *,
        skips: Iterable[QueryID] = (),
        **kwds: Any,
    ) -> None:
        self.name = name
        impl = nw.Implementation.from_backend(into_backend)
        assert impl is not nw.Implementation.UNKNOWN
        self.implementation = impl
        self.skips = frozenset(skips)
        self.kwds = kwds

    def __repr__(self) -> str:
        return self.name

    def scan(self, source: FileSource) -> nw.LazyFrame[Any]:
        return nw.scan_parquet(source, backend=self.implementation, **self.kwds)


class Query:
    id: QueryID
    paths: tuple[Path, ...]
    PACKAGE_PREFIX: ClassVar = "tpch.queries"

    def __init__(self, query_id: QueryID, paths: tuple[Path, ...]) -> None:
        self.id = query_id
        self.paths = paths

    def __repr__(self) -> str:
        return self.id

    def _import_module(self) -> QueryModule:
        result: Any = import_module(f"{self.PACKAGE_PREFIX}.{self}")
        return result

    def expected(self) -> pl.DataFrame:
        return pl.read_parquet(DATA_DIR / f"result_{self}.parquet")

    def run(self, backend: Backend) -> pl.DataFrame:
        if self.id in backend.skips:
            pytest.skip(f"Query {self} is not supported for {backend}")
        data = (backend.scan(fp.as_posix()) for fp in self.paths)
        return self._import_module().query(*data).lazy().collect("polars").to_polars()


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
        # NOTE: https://github.com/narwhals-dev/narwhals/issues/2226
        yield Backend("duckdb", "duckdb", skips=["q15"])
        if find_spec("sqlframe"):
            from sqlframe.duckdb import DuckDBSession

            yield Backend("sqlframe", "sqlframe", skips=["q15"], session=DuckDBSession())


@pytest.fixture(params=iter_backends(), ids=repr)
def backend(request: pytest.FixtureRequest) -> Backend:
    result: Backend = request.param
    return result


def q(query_id: QueryID, *paths: Path) -> Query:
    return Query(query_id, paths)


queries = (
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
    q("q17", LINEITEM_PATH, PART_PATH),
    q("q18", CUSTOMER_PATH, LINEITEM_PATH, ORDERS_PATH),
    q("q19", LINEITEM_PATH, PART_PATH),
    q("q20", PART_PATH, PARTSUPP_PATH, NATION_PATH, LINEITEM_PATH, SUPPLIER_PATH),
    q("q21", LINEITEM_PATH, NATION_PATH, ORDERS_PATH, SUPPLIER_PATH),
    q("q22", CUSTOMER_PATH, ORDERS_PATH),
)


@pytest.fixture(params=queries, ids=repr)
def query(request: pytest.FixtureRequest) -> Query:
    result: Query = request.param
    return result


@pytest.fixture(scope="session")
def generate_data_metadata() -> pl.DataFrame:
    from tpch.generate_data import METADATA_PATH

    return pl.read_csv(METADATA_PATH)


@pytest.fixture(scope="session")
def scale_factor(generate_data_metadata: pl.DataFrame) -> float:
    return float(generate_data_metadata.get_column("scale_factor").item())
