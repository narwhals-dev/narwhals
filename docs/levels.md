# Levels

Narwhals comes with two levels of support: "full" and "interchange".

Libraries for which we have full support can benefit from the whole
[Narwhals API](https://narwhals-dev.github.io/narwhals/api-reference/).

For example:

```python exec="1" source="above"
import narwhals as nw
from narwhals.typing import FrameT

@nw.narwhalify
def func(df: FrameT) -> FrameT:
    return df.group_by('a').agg(
        b_mean=nw.col('b').mean(),
        b_std=nw.col('b').std(),
    )
```
will work for any of pandas, Polars, cuDF, and Modin.

However, sometimes you don't need to do complex operations on dataframes - all you need
is to inspect the schema a bit before making other decisions, such as which columns to
select or whether to convert to another library. For that purpose, we also provide "interchange"
level of support. If a library implements the
[Dataframe Interchange Protocol](https://data-apis.org/dataframe-protocol/latest/), then
a call such as

```python exec="1" source="above"
from typing import Any

import narwhals as nw
from narwhals.schema import Schema


def func(df_any: Any) -> Schema:
    df = nw.from_native(df, eager_or_interchange_only=True)
    return df.schema
```
is also supported, meaning that, in addition to the libraries mentioned above, you can
also pass Ibis, Vaex, PyArrow, and any other library which implements the protocol.
