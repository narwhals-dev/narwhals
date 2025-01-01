from __future__ import annotations

from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

if not Path("data").exists():
    Path("data").mkdir()

con = duckdb.connect(database=":memory:")
con.execute("INSTALL tpch; LOAD tpch")
con.execute("CALL dbgen(sf=.5)")
tables = [
    "lineitem",
    "customer",
    "nation",
    "orders",
    "part",
    "partsupp",
    "region",
    "supplier",
]
for t in tables:
    res = con.query("SELECT * FROM " + t)  # noqa: S608
    res_arrow = res.to_arrow_table()
    new_schema = []
    for field in res_arrow.schema:
        if isinstance(field.type, type(pa.decimal128(1))):
            new_schema.append(pa.field(field.name, pa.float64()))
        elif field.type == pa.date32():
            new_schema.append(pa.field(field.name, pa.timestamp("ns")))
        else:
            new_schema.append(field)
    res_arrow = res_arrow.cast(pa.schema(new_schema))
    pq.write_table(res_arrow, Path("data") / f"{t}.parquet")
