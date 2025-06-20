from __future__ import annotations

import io
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.csv as pc
import pyarrow.parquet as pq

Path("data").mkdir(exist_ok=True)

SCALE_FACTOR = 0.1
con = duckdb.connect(database=":memory:")
con.execute("INSTALL tpch; LOAD tpch")
con.execute(f"CALL dbgen(sf={SCALE_FACTOR})")
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


def convert_schema(schema: pa.Schema) -> pa.Schema:
    new_schema = []
    for field in schema:
        if pa.types.is_decimal(field.type):
            new_schema.append(pa.field(field.name, pa.float64()))
        elif field.type == pa.date32():
            new_schema.append(pa.field(field.name, pa.timestamp("ns")))
        else:
            new_schema.append(field)
    return pa.schema(new_schema)


for t in tables:
    tbl = con.query(f"SELECT * FROM {t}")  # noqa: S608
    tbl_arrow = tbl.to_arrow_table()
    new_schema = convert_schema(tbl_arrow.schema)
    tbl_arrow = tbl_arrow.cast(new_schema)
    pq.write_table(tbl_arrow, Path("data") / f"{t}.parquet")


results = con.query(
    f"""
    SELECT query_nr, answer
    FROM tpch_answers()
    WHERE scale_factor={SCALE_FACTOR}
"""  # noqa: S608
)

while row := results.fetchmany(1):
    query_nr, answer = row[0]
    tbl_answer = pc.read_csv(
        io.BytesIO(answer.encode("utf-8")), parse_options=pc.ParseOptions(delimiter="|")
    )
    new_schema = convert_schema(tbl_answer.schema)
    tbl_answer = tbl_answer.cast(new_schema)

    pq.write_table(tbl_answer, Path("data") / f"result_q{query_nr}.parquet")
