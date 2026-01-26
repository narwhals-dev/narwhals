from __future__ import annotations

# ruff: noqa: S608
import io
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal, get_args

from tpch.typing_ import QueryID

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

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import pyarrow as pa
    from duckdb import DuckDBPyConnection
    from typing_extensions import TypeAlias

    from narwhals.typing import SizeUnit

    BuiltinScaleFactor: TypeAlias = Literal["0.01", "0.1", "1.0"]
    FileName: TypeAlias = str
    FileSize: TypeAlias = float


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

# Required after each call to `dbgen`, if persisting the database
Q_CLEANUP = """
DROP TABLE IF EXISTS customer;
DROP TABLE IF EXISTS lineitem;
DROP TABLE IF EXISTS nation;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS part;
DROP TABLE IF EXISTS partsupp;
DROP TABLE IF EXISTS region;
DROP TABLE IF EXISTS supplier;
"""

SF_BUILTIN_STR: Mapping[float, BuiltinScaleFactor] = {
    0.01: "0.01",
    0.1: "0.1",
    1.0: "1.0",
}


def format_size(n_bytes: int) -> tuple[FileSize, SizeUnit]:
    """Return the best human-readable size and unit for the given byte count."""
    units = ("b", "kb", "mb", "gb", "tb")
    size = float(n_bytes)
    for unit in units:
        if size < 1024 or unit == "tb":
            return size, unit
        size /= 1024
    return size, "tb"


class TableLogger:
    """A logger that streams table rows with box-drawing characters."""

    # Size column: 3 leading digits + 1 dot + 2 decimals + 1 space + 2 unit chars = 10 chars
    SIZE_WIDTH = 9

    def __init__(self, file_names: Sequence[FileName]) -> None:
        self._file_width = max(len(name) for name in file_names)
        self._started = False

    def _log_header(self) -> None:
        fw, sw = self._file_width, self.SIZE_WIDTH

        logger.info("┌─%s─┬─%s─┐", "─" * fw, "─" * sw)
        logger.info("│ %s ┆ %s │", "File".rjust(fw), "Size".rjust(sw))
        logger.info("╞═%s═╪═%s═╡", "═" * fw, "═" * sw)

    def log_row(self, name: FileName, n_bytes: int) -> None:
        if not self._started:
            self._log_header()
            self._started = True

        size, unit = format_size(n_bytes)
        size_str = f"{size:>6.2f} {unit:>2}"
        logger.info("│ %s ┆ %s │", name.rjust(self._file_width), size_str)

    def log_footer(self) -> None:
        if self._started:
            fw, sw = self._file_width, self.SIZE_WIDTH
            logger.info("└─%s─┴─%s─┘", "─" * fw, "─" * sw)


def answers_any(con: DuckDBPyConnection) -> None:
    import pyarrow.parquet as pq

    query_ids = get_args(QueryID)
    file_names = tuple(f"result_{qid}.parquet" for qid in query_ids)

    logger.info("Executing tpch queries for answers")
    tbl_logger = TableLogger(file_names)

    for query_id in query_ids:
        query_num = str(query_id).removeprefix("q")
        result = con.sql(f"PRAGMA tpch({query_num})")
        result_pa = result.to_arrow_table()
        result_pa = result_pa.cast(convert_schema(result_pa.schema))
        path = DATA / f"result_{query_id}.parquet"
        pq.write_table(result_pa, path)
        tbl_logger.log_row(path.name, result_pa.nbytes)

    tbl_logger.log_footer()


def answers_builtin(con: DuckDBPyConnection, scale: BuiltinScaleFactor) -> None:
    import pyarrow.csv as pc
    import pyarrow.parquet as pq

    # Pre-compute file names for table width (queries 1-22)
    file_names = tuple(f"result_q{i}.parquet" for i in range(1, 23))

    logger.info("Fastpath for builtin tpch_answers()")
    tbl_logger = TableLogger(file_names)

    results = con.sql(
        f"""
        SELECT query_nr, answer
        FROM tpch_answers()
        WHERE scale_factor={scale}
        """
    )
    while row := results.fetchmany(1):
        query_nr, answer = row[0]
        tbl_answer = pc.read_csv(
            io.BytesIO(answer.encode("utf-8")),
            parse_options=pc.ParseOptions(delimiter="|"),
        )
        tbl_answer = tbl_answer.cast(convert_schema(tbl_answer.schema))
        path = DATA / f"result_q{query_nr}.parquet"
        pq.write_table(tbl_answer, path)
        tbl_logger.log_row(path.name, tbl_answer.nbytes)

    tbl_logger.log_footer()


def load_tpch(con: DuckDBPyConnection) -> None:
    logger.info("Installing DuckDB TPC-H Extension")
    con.install_extension("tpch")
    con.load_extension("tpch")


def convert_schema(schema: pa.Schema) -> pa.Schema:
    import pyarrow as pa

    new_schema = []
    for field in schema:
        if pa.types.is_decimal(field.type):
            new_schema.append(pa.field(field.name, pa.float64()))
        elif field.type == pa.date32():
            new_schema.append(pa.field(field.name, pa.timestamp("ns")))
        else:
            new_schema.append(field)
    return pa.schema(new_schema)


def main(scale_factor: float = 0.1) -> None:
    import duckdb
    import pyarrow.parquet as pq

    DATA.mkdir(exist_ok=True)
    logger.info("Connecting to in-memory DuckDB database")
    con = duckdb.connect(database=":memory:")
    load_tpch(con)
    logger.info("Generating data with scale_factor=%s", scale_factor)
    con.sql(f"CALL dbgen(sf={scale_factor})")
    logger.info("Finished generating data.")

    tables = (
        "lineitem",
        "customer",
        "nation",
        "orders",
        "part",
        "partsupp",
        "region",
        "supplier",
    )
    file_names = tuple(f"{t}.parquet" for t in tables)

    logger.info("Writing data to: %s", DATA.as_posix())
    tbl_logger = TableLogger(file_names)
    for t in tables:
        tbl = con.sql(f"SELECT * FROM {t}")
        tbl_arrow = tbl.to_arrow_table()
        tbl_arrow = tbl_arrow.cast(convert_schema(tbl_arrow.schema))
        path = DATA / f"{t}.parquet"
        pq.write_table(tbl_arrow, path)
        tbl_logger.log_row(path.name, tbl_arrow.nbytes)
    tbl_logger.log_footer()

    logger.info("Getting answers")
    if scale := SF_BUILTIN_STR.get(scale_factor):
        answers_builtin(con, scale)
    else:
        answers_any(con)


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
