from __future__ import annotations

# ruff: noqa: S608
import io
import logging
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Literal, get_args

from tpch.typing_ import QueryID

if TYPE_CHECKING:
    from typing_extensions import Self

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
    from collections.abc import Iterable, Mapping

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
        size, unit = format_size(n_bytes)
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


@cache
def query_ids() -> tuple[str, ...]:
    return get_args(QueryID)


def answers_any(con: DuckDBPyConnection) -> None:
    import pyarrow.parquet as pq

    logger.info("Executing tpch queries for answers")
    with TableLogger.answers() as tbl_logger:
        for query_id in query_ids():
            result = con.sql(f"PRAGMA tpch({query_id.removeprefix('q')})")
            result_pa = result.to_arrow_table()
            result_pa = result_pa.cast(convert_schema(result_pa.schema))
            path = DATA / f"result_{query_id}.parquet"
            pq.write_table(result_pa, path)
            tbl_logger.log_row(path.name, result_pa.nbytes)


def answers_builtin(con: DuckDBPyConnection, scale: BuiltinScaleFactor) -> None:
    import pyarrow.csv as pc
    import pyarrow.parquet as pq

    logger.info("Fastpath for builtin tpch_answers()")
    results = con.sql(
        f"""
            SELECT query_nr, answer
            FROM tpch_answers()
            WHERE scale_factor={scale}
            """
    )
    with TableLogger.answers() as tbl_logger:
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
    logger.info("Writing data to: %s", DATA.as_posix())
    with TableLogger.sources() as tbl_logger:
        for t in SOURCES:
            tbl = con.sql(f"SELECT * FROM {t}")
            tbl_arrow = tbl.to_arrow_table()
            tbl_arrow = tbl_arrow.cast(convert_schema(tbl_arrow.schema))
            path = DATA / f"{t}.parquet"
            pq.write_table(tbl_arrow, path)
            tbl_logger.log_row(path.name, tbl_arrow.nbytes)
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
