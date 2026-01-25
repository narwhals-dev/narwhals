from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING

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


def main(scale_factor: float = 0.1) -> None:
    import duckdb
    import pyarrow.csv as pc
    import pyarrow.parquet as pq

    data_path = Path("data")
    data_path.mkdir(exist_ok=True)
    con = duckdb.connect(database=":memory:")
    con.execute("INSTALL tpch; LOAD tpch")
    con.execute(f"CALL dbgen(sf={scale_factor})")
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
        pq.write_table(tbl_arrow, data_path / f"{t}.parquet")

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

        pq.write_table(tbl_answer, data_path / f"result_q{query_nr}.parquet")


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
