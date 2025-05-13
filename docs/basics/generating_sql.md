# Generating SQL

You can use Narwhals as a lightweight tool to generate SQL queries.

For example:

```python exec="1" source="above" session="sql-generation" result="python"
import narwhals as nw
from narwhals.typing import IntoFrameT


def monthly_aggregate(
    df_native: IntoFrameT,
    date_column: str,
    price_column: str,
) -> IntoFrameT:
    return (
        nw.from_native(df_native)
        .group_by(nw.col(date_column).dt.truncate("1mo"))
        .agg(nw.col(price_column).mean())
        .sort(date_column)
        .to_native()
    )


assets = nw.sql.table(
    "assets", {"date": nw.Date, "price": nw.Int64, "symbol": nw.String}
)
print(monthly_aggregate(assets, "date", "price").sql(dialect="duckdb"))
```

This uses [SQLFrame](https://github.com/eakmanrq/sqlframe) under the hood, so you're
required to have that installed too. You can install required SQL dependencies
with:

```console
pip install -U narwhals[sql]
```
