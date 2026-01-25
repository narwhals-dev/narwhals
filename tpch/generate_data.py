from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from narwhals._utils import scale_bytes

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
    import pyarrow as pa

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
    import pyarrow.csv as pc
    import pyarrow.parquet as pq

    DATA.mkdir(exist_ok=True)
    logger.info("Connecting to in-memory DuckDB database")
    con = duckdb.connect(database=":memory:")
    logger.info("Installing DuckDB TPC-H Extension")
    con.execute("INSTALL tpch; LOAD tpch")
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
        tbl = con.query(f"SELECT * FROM {t}")  # noqa: S608
        tbl_arrow = tbl.to_arrow_table()
        new_schema = convert_schema(tbl_arrow.schema)
        tbl_arrow = tbl_arrow.cast(new_schema)
        path = DATA / f"{t}.parquet"
        log_write(path, tbl_arrow.nbytes, "mb")
        pq.write_table(tbl_arrow, path)

    logger.info("Getting answers")
    # TODO @dangotbanned: Use `PRAGMA tpch(query_id);` when `scale_factor not in {0.01, 0.1, 1}`
    results = con.query(
        f"""
        SELECT query_nr, answer
        FROM tpch_answers()
        WHERE scale_factor={scale_factor}
        """  # noqa: S608
    )

    while row := results.fetchmany(1):
        query_nr, answer = row[0]
        tbl_answer = pc.read_csv(
            io.BytesIO(answer.encode("utf-8")),
            parse_options=pc.ParseOptions(delimiter="|"),
        )
        new_schema = convert_schema(tbl_answer.schema)
        tbl_answer = tbl_answer.cast(new_schema)
        path = DATA / f"result_q{query_nr}.parquet"
        log_write(path, tbl_answer.nbytes, "b")
        pq.write_table(tbl_answer, path)


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
