# Supported libraries and extending Narwhals

## List of supported libraries (and how to add yours!)

Currently, Narwhals has **full API** support for the following libraries:

| Library  | 🔗 Link 🔗 |
| ------------- | ------------- |
| ️Polars 🐻‍❄️ | [github.com/pola-rs/polars](https://github.com/pola-rs/polars) |
| pandas 🐼 |  [github.com/pandas-dev/pandas](https://github.com/pandas-dev/pandas) |
| cuDF | [github.com/rapidsai/cudf](https://github.com/rapidsai/cudf) |
| Modin | [github.com/modin-project/modin](https://github.com/modin-project/modin) |
| PyArrow ⇶ | [arrow.apache.org/docs/python](https://arrow.apache.org/docs/python/index.html) |

It also has **lazy-only** support for [Dask](https://github.com/dask/dask), and **interchange** support
for [DuckDB](https://github.com/duckdb/duckdb) and [Ibis](https://github.com/ibis-project/ibis).

We are working towards full "lazy-only" support for DuckDB, Ibis, and PySpark.

### Levels of support

Narwhals comes with three levels of support:

- **Full API support**: cuDF, Modin, pandas, Polars, PyArrow
- **Lazy-only support**: Dask. Work in progress: DuckDB, Ibis, PySpark.
- **Interchange-level support**: DuckDB, Ibis, Vaex, anything which implements the DataFrame Interchange Protocol

Libraries for which we have full support can benefit from the whole
[Narwhals API](./api-reference/index.md).

For example:

=== "from/to_native"
    ```python exec="1" source="above"
    import narwhals as nw
    from narwhals.typing import IntoFrameT


    def func(df: IntoFrameT) -> IntoFrameT:
        return (
            nw.from_native(df)
            .group_by("a")
            .agg(
                b_mean=nw.col("b").mean(),
                b_std=nw.col("b").std(),
            )
            .to_native()
        )
    ```

=== "@narwhalify"
    ```python exec="1" source="above"
    import narwhals as nw
    from narwhals.typing import FrameT


    @nw.narwhalify
    def func(df: FrameT) -> FrameT:
        return df.group_by("a").agg(
            b_mean=nw.col("b").mean(),
            b_std=nw.col("b").std(),
        )
    ```

will work for any of pandas, Polars, cuDF, Modin, and PyArrow.

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


def func(df: Any) -> Schema:
    df = nw.from_native(df, eager_or_interchange_only=True)
    return df.schema
```

is also supported, meaning that, in addition to the libraries mentioned above, you can
also pass Ibis, DuckDB, Vaex, and any library which implements the protocol.

#### Interchange-only support

While libraries for which we have full support can benefit from the whole Narwhals API,
libraries which have interchange only support can access the following methods after 
converting to Narwhals DataFrame:

- `.schema`, hence column names via `.schema.names()` and column types via `.schema.dtypes()`
- `.columns`
- `.to_pandas()` and `.to_arrow()`, for converting to Pandas and Arrow, respectively.
- `.select(names)` (Ibis and DuckDB), where `names` is a list of (string) column names. This is useful for
  selecting columns before converting to another library.

## Extending Narwhals

If you want your own library to be recognised too, you're welcome open a PR (with tests)!.
Alternatively, if you can't do that (for example, if you library is closed-source), see
the next section for what else you can do.

We love open source, but we're not "open source absolutists". If you're unable to open
source you library, then this is how you can make your library compatible with Narwhals.

Make sure that you also define:

  - `DataFrame.__narwhals_dataframe__`: return an object which implements methods from the
    `CompliantDataFrame` protocol in  `narwhals/typing.py`.
  - `DataFrame.__narwhals_namespace__`: return an object which implements methods from the
    `CompliantNamespace` protocol in `narwhals/typing.py`.
  - `DataFrame.__native_namespace__`: return the object's native namespace.
  - `LazyFrame.__narwhals_lazyframe__`: return an object which implements methods from the
    `CompliantLazyFrame` protocol in  `narwhals/typing.py`.
  - `LazyFrame.__narwhals_namespace__`: return an object which implements methods from the
    `CompliantNamespace` protocol in `narwhals/typing.py`.
  - `LazyFrame.__native_namespace__`: return the object's native namespace.
  - `Series.__narwhals_series__`: return an object which implements methods from the
    `CompliantSeries` protocol in `narwhals/typing.py`.

  If your library doesn't distinguish between lazy and eager, then it's OK for your dataframe
  object to implement both `__narwhals_dataframe__` and `__narwhals_lazyframe__`. In fact,
  that's currently what `narwhals._pandas_like.dataframe.PandasLikeDataFrame` does. So, if you're stuck,
  take a look at the source code to see how it's done!

Note that this "extension" mechanism is still experimental. If anything is not clear, or
doesn't work, please do raise an issue or contact us on Discord (see the link on the README).
