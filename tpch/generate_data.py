from __future__ import annotations

import io
import logging
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Literal, get_args

from tpch.typing_ import QueryID

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    import pyarrow as pa
    from duckdb import DuckDBPyConnection as Con
    from typing_extensions import Self, TypeAlias

    from narwhals.typing import SizeUnit

    BuiltinScaleFactor: TypeAlias = Literal["0.01", "0.1", "1.0"]
    FileName: TypeAlias = str
    FileSize: TypeAlias = float

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
log_output = logging.StreamHandler()
log_output.setFormatter(
    logging.Formatter(
        "%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"
    )
)
logger.addHandler(log_output)

REPO_ROOT = Path(__file__).parent.parent
TPCH_ROOT = REPO_ROOT / "tpch"
DATA = TPCH_ROOT / "data"


TABLE_SCALE_FACTOR = """
┌──────────────┬───────────────┐
│ Scale factor ┆ Database (MB) │
╞══════════════╪═══════════════╡
│ 0.1          ┆ 25            │
│ 1.0          ┆ 250           │
│ 3.0          ┆ 754           │
│ 100.0        ┆ 26624         │
└──────────────┴───────────────┘
"""


SF_BUILTIN_STR: Mapping[float, BuiltinScaleFactor] = {
    0.01: "0.01",
    0.1: "0.1",
    1.0: "1.0",
}

SOURCES = (
    "lineitem",
    "customer",
    "nation",
    "orders",
    "part",
    "partsupp",
    "region",
    "supplier",
)

# NOTE: Store queries here, add parameter names if needed
SQL_DBGEN = "CALL dbgen(sf={0})"
SQL_TPCH_ANSWER = "PRAGMA tpch({0})"
SQL_TPCH_ANSWERS = """
SELECT query_nr, answer
FROM tpch_answers()
WHERE scale_factor={0}
"""
SQL_FROM = "FROM {0}"


@cache
def query_ids() -> tuple[str, ...]:
    return get_args(QueryID)


def _downcast_exotic_types(table: pa.Table) -> pa.Table:
    import pyarrow as pa

    new_schema = []
    for field in table.schema:
        if pa.types.is_decimal(field.type):
            new_schema.append(pa.field(field.name, pa.float64()))
        elif field.type == pa.date32():
            new_schema.append(pa.field(field.name, pa.timestamp("ns")))
        else:
            new_schema.append(field)
    return table.cast(pa.schema(new_schema))


class TableLogger:
    """A logger that streams table rows with box-drawing characters."""

    # Size column: 3 leading digits + 1 dot + 2 decimals + 1 space + 2 unit chars = 10 chars
    SIZE_WIDTH = 9

    def __init__(self, file_names: Iterable[FileName]) -> None:
        self._file_width = max(len(name) for name in file_names)

    @staticmethod
    def sources() -> TableLogger:
        return TableLogger(f"{t}.parquet" for t in SOURCES)

    @staticmethod
    def answers() -> TableLogger:
        return TableLogger(f"result_{qid}.parquet" for qid in query_ids())

    def __enter__(self) -> Self:
        self._log_header()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self._log_footer()

    def log_row(self, name: FileName, n_bytes: int) -> None:
        size, unit = self._format_size(n_bytes)
        size_str = f"{size:>6.2f} {unit:>2}"
        logger.info("│ %s ┆ %s │", name.rjust(self._file_width), size_str)

    def _log_header(self) -> None:
        fw, sw = self._file_width, self.SIZE_WIDTH
        logger.info("┌─%s─┬─%s─┐", "─" * fw, "─" * sw)
        logger.info("│ %s ┆ %s │", "File".rjust(fw), "Size".rjust(sw))
        logger.info("╞═%s═╪═%s═╡", "═" * fw, "═" * sw)

    def _log_footer(self) -> None:
        fw, sw = self._file_width, self.SIZE_WIDTH
        logger.info("└─%s─┴─%s─┘", "─" * fw, "─" * sw)

    @staticmethod
    def _format_size(n_bytes: int) -> tuple[FileSize, SizeUnit]:
        """Return the best human-readable size and unit for the given byte count."""
        units = ("b", "kb", "mb", "gb", "tb")
        size = float(n_bytes)
        for unit in units:
            if size < 1024 or unit == "tb":
                return size, unit
            size /= 1024
        return size, "tb"


def connect() -> Con:
    import duckdb

    logger.info("Connecting to in-memory DuckDB database")
    return duckdb.connect(database=":memory:")


def load_tpch_extension(con: Con) -> Con:
    logger.info("Installing DuckDB TPC-H Extension")
    con.install_extension("tpch")
    con.load_extension("tpch")
    return con


def generate_tpch_database(con: Con, scale_factor: float) -> Con:
    logger.info("Generating data with scale_factor=%s", scale_factor)
    con.sql(SQL_DBGEN.format(scale_factor))
    logger.info("Finished generating data.")
    return con


def write_tpch_database(con: Con) -> Con:
    import pyarrow.parquet as pq

    logger.info("Writing data to: %s", DATA.as_posix())
    with TableLogger.sources() as tbl_logger:
        for t in SOURCES:
            table = _downcast_exotic_types(con.sql(SQL_FROM.format(t)).to_arrow_table())
            path = DATA / f"{t}.parquet"
            pq.write_table(table, path)
            tbl_logger.log_row(path.name, table.nbytes)
    return con


def _answers_any(con: Con) -> Con:
    import pyarrow.parquet as pq

    logger.info("Executing tpch queries for answers")
    with TableLogger.answers() as tbl_logger:
        for query_id in query_ids():
            query = SQL_TPCH_ANSWER.format(query_id.removeprefix("q"))
            table = _downcast_exotic_types(con.sql(query).to_arrow_table())
            path = DATA / f"result_{query_id}.parquet"
            pq.write_table(table, path)
            tbl_logger.log_row(path.name, table.nbytes)
    return con


def _answers_builtin(con: Con, scale: BuiltinScaleFactor) -> Con:
    import pyarrow.parquet as pq
    from pyarrow.csv import ParseOptions, read_csv

    logger.info("Fastpath for builtin tpch_answers()")
    results = con.sql(SQL_TPCH_ANSWERS.format(scale))
    opts = ParseOptions(delimiter="|")
    with TableLogger.answers() as tbl_logger:
        while row := results.fetchmany(1):
            query_nr, answer = row[0]
            table = read_csv(io.BytesIO(answer.encode("utf-8")), parse_options=opts)
            table = _downcast_exotic_types(table)
            path = DATA / f"result_q{query_nr}.parquet"
            pq.write_table(table, path)
            tbl_logger.log_row(path.name, table.nbytes)
    return con


def write_tpch_answers(con: Con, scale_factor: float) -> Con:
    logger.info("Getting answers")
    if scale := SF_BUILTIN_STR.get(scale_factor):
        return _answers_builtin(con, scale)
    return _answers_any(con)


def main(scale_factor: float = 0.1) -> None:
    DATA.mkdir(exist_ok=True)
    con = connect()
    load_tpch_extension(con)
    generate_tpch_database(con, scale_factor)
    write_tpch_database(con)
    write_tpch_answers(con, scale_factor)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Generate the data required to run TPCH queries.",
    )
    parser.add_argument(
        "-sf",
        "--scale-factor",
        default="0.1",
        dest="scale_factor",
        help=f"Scale the database by this factor (default: %(default)s)\n{TABLE_SCALE_FACTOR}",
        type=float,
    )
    args = parser.parse_args()
    main(scale_factor=args.scale_factor)
