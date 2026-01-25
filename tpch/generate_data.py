from __future__ import annotations

# ruff: noqa: S608
import io
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal, get_args

from narwhals._utils import scale_bytes
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
    from collections.abc import Mapping

    import pyarrow as pa
    from duckdb import DuckDBPyConnection
    from typing_extensions import TypeAlias

BuiltinScaleFactor: TypeAlias = Literal["0.01", "0.1", "1.0"]

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


def answers_any(con: DuckDBPyConnection) -> None:
    import pyarrow.parquet as pq

    logger.info("Executing tpch queries for answers")

    for query_id in get_args(QueryID):
        query_num = str(query_id).removeprefix("q")
        result = con.sql(f"PRAGMA tpch({query_num})")
        result_pa = result.to_arrow_table()
        result_pa = result_pa.cast(convert_schema(result_pa.schema))
        path = DATA / f"result_{query_id}.parquet"
        log_write(path, result_pa.nbytes, "b")
        pq.write_table(result_pa, path)


def answers_builtin(con: DuckDBPyConnection, scale: BuiltinScaleFactor) -> None:
    import pyarrow.csv as pc
    import pyarrow.parquet as pq

    logger.info("Fastpath for builtin tpch_answers() where scale_factor=%s", scale)

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
        log_write(path, tbl_answer.nbytes, "b")
        pq.write_table(tbl_answer, path)


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


def log_write(path: Path, n_bytes: int, unit: Literal["b", "kb", "mb"]) -> None:
    size = float(n_bytes) if unit == "b" else scale_bytes(n_bytes, unit)
    logger.info("Writing % 20.4f %s -> %s", size, unit, path.as_posix())


def main(scale_factor: float = 0.1) -> None:
    import duckdb
    import pyarrow.parquet as pq

    DATA.mkdir(exist_ok=True)
    logger.info("Connecting to in-memory DuckDB database")
    con = duckdb.connect(database=":memory:")
    logger.info("Installing DuckDB TPC-H Extension")
    con.load_extension("tpch")
    logger.info("Generating data for `scale_factor=%s`", scale_factor)
    con.execute(f"CALL dbgen(sf={scale_factor})")
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
    for t in tables:
        tbl = con.query(f"SELECT * FROM {t}")
        tbl_arrow = tbl.to_arrow_table()
        tbl_arrow = tbl_arrow.cast(convert_schema(tbl_arrow.schema))
        path = DATA / f"{t}.parquet"
        log_write(path, tbl_arrow.nbytes, "mb")
        pq.write_table(tbl_arrow, path)

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
