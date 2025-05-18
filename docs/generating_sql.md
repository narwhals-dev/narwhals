# Generating SQL

Suppose you want to write Polars syntax and translate it to SQL.
For example, what's the SQL equivalent to:

```python exec="1" source="above" session="generating-sql"
import narwhals as nw
from narwhals.typing import IntoFrameT


def avg_monthly_price(df_native: IntoFrameT) -> IntoFrameT:
    return (
        nw.from_native(df_native)
        .group_by(nw.col("date").dt.truncate("1mo"))
        .agg(nw.col("price").mean())
        .sort("date")
        .to_native()
    )
```

?

There are several ways to find out.

## Via SQLFrame (most lightweight solution)

The most lightweight solution which does not require any heavy dependencies, nor
any actual table or dataframe, is with SQLFrame.

```python exec="1" source="above" session="generating-sql" result="sql"
from sqlframe.standalone import StandaloneSession

session = StandaloneSession.builder.getOrCreate()
session.catalog.add_table("prices", column_mapping={"date": "date", "price": "float"})
df = nw.from_native(session.read.table("prices"))

print(avg_monthly_price(df).sql(dialect="duckdb"))
```

Or, to print the SQL code in a different dialect (say, databricks):

```python exec="1" source="above" session="generating-sql" result="sql"
print(avg_monthly_price(df).sql(dialect="duckdb"))
```

## Via DuckDB

You can also generate SQL directly from DuckDB.

```python exec="1" source="above" session="generating-sql" result="sql"
import duckdb

conn = duckdb.connect()
conn.sql("""CREATE TABLE prices (date DATE, price DOUBLE);""")

df = nw.from_native(conn.table("prices"))
print(avg_monthly_price(df).sql_query())
```

To make it look a bit prettier, we can pass it to [SQLGlot](https://github.com/tobymao/sqlglot):

```python exec="1" source="above" session="generating-sql" result="sql"
import sqlglot

print(sqlglot.transpile(avg_monthly_price(df).sql_query(), pretty=True)[0])
```

## Via Ibis

We can also use Ibis to generate SQL:

```python exec="1" source="above" session="generating-sql" result="sql"
import ibis

t = ibis.table({"date": "date", "price": "double"}, name="prices")
print(ibis.to_sql(avg_monthly_price(t)))
```
