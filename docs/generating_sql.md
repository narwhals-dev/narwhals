# Generating SQL

Suppose you want to write Polars syntax and translate it to SQL.
For example, what's the SQL equivalent to:

```python exec="1" source="above" session="generating-sql"
import narwhals as nw
from narwhals.typing import FrameT


def avg_monthly_price(df: FrameT) -> FrameT:
    return (
        df.group_by(nw.col("date").dt.truncate("1mo"))
        .agg(nw.col("price").mean())
        .sort("date")
    )
```

?

Narwhals provides you with a `narwhals.sql` module to do just that!

!!! info
    `narwhals.sql` currently requires DuckDB to be installed.

## `narwhals.sql`

You can generate SQL directly from DuckDB.

```python exec="1" source="above" session="generating-sql" result="sql"
import narwhals as nw
from narwhals.sql import table

prices = table("prices", {"date": nw.Date, "price": nw.Float64})

result = (
    prices.group_by(nw.col("date").dt.truncate("1mo"))
    .agg(nw.col("price").mean())
    .sort("date")
)
print(result.to_sql())
```

To make it look a bit prettier, you can pass `pretty=True`, but
note that this currently requires [sqlparse](https://github.com/andialbrecht/sqlparse) to be installed.

```python exec="1" source="above" session="generating-sql" result="sql"
print(result.to_sql(pretty=True))
```

## Via Ibis

You can also use Ibis or SQLFrame to generate SQL:

```python exec="1" source="above" session="generating-sql" result="sql"
import ibis

df = nw.from_native(ibis.table({"date": "date", "price": "double"}, name="prices"))
print(ibis.to_sql(avg_monthly_price(df).to_native()))
```

## Via SQLFrame

You can also use SQLFrame:

```python exec="1" source="above" session="generating-sql" result="sql"
from sqlframe.standalone import StandaloneSession

session = StandaloneSession.builder.getOrCreate()
session.catalog.add_table("prices", column_mapping={"date": "date", "price": "float"})
df = nw.from_native(session.read.table("prices"))

print(avg_monthly_price(df).to_native().sql(dialect="duckdb"))
```
