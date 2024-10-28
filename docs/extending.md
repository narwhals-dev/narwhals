# Extending Narwhals, levels of support

## List of supported libraries (and how to add yours!)

Currently, Narwhals has full API support for the following libraries:

| Library  | 🔗 Link 🔗 |
| ------------- | ------------- |
| ️Polars 🐻‍❄️ | [github.com/pola-rs/polars](https://github.com/pola-rs/polars) |
| pandas 🐼 |  [github.com/pandas-dev/pandas](https://github.com/pandas-dev/pandas) |
| cuDF | [github.com/rapidsai/cudf](https://github.com/rapidsai/cudf) |
| Modin | [github.com/modin-project/modin](https://github.com/modin-project/modin) |
| PyArrow ⇶ | [arrow.apache.org/docs/python](https://arrow.apache.org/docs/python/index.html) |

It also has lazy-only support for [Dask](https://github.com/dask/dask), and interchange-only support
for [DuckDB](https://github.com/duckdb/duckdb) and [Ibis](https://github.com/ibis-project/ibis).

### Levels

Narwhals comes with two levels of support ("full" and "interchange"), and we are working on defining
a "lazy-only" level too.

Libraries for which we have full support can benefit from the whole
[Narwhals API](https://narwhals-dev.github.io/narwhals/api-reference/).

For example:

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
libraries which only benefit from interchange support can only access the following methods:

- `.schema`, hence column names via `.schema.names()` and column types via `.schema.dtypes()`
- `.to_pandas()` and `.to_arrow()`, for converting to Pandas and Arrow, respectively.
- `.select(names)` (Ibis and DuckDB), where `names` is a list of (string) column names. This is useful for
  selecting columns before converting to another library.

### Extending Narwhals

If you want your own library to be recognised too, you're welcome open a PR (with tests)!.
Alternatively, if you can't do that (for example, if you library is closed-source), see
the next section for what else you can do.

We love open source, but we're not "open source absolutists". If you're unable to open
source you library, then this is how you can make your library compatible with Narwhals.

Make sure that, in addition to the public Narwhals API, you also define:

  - `DataFrame.__narwhals_dataframe__`: return an object which implements public methods
    from `Narwhals.DataFrame`
  - `DataFrame.__narwhals_namespace__`: return an object which implements public top-level
    functions from `narwhals` (e.g. `narwhals.col`, `narwhals.concat`, ...)
  - `DataFrame.__native_namespace__`: return a native namespace object which must have a
    `from_dict` method
  - `LazyFrame.__narwhals_lazyframe__`: return an object which implements public methods
    from `Narwhals.LazyFrame`
  - `LazyFrame.__narwhals_namespace__`: return an object which implements public top-level
    functions from `narwhals` (e.g. `narwhals.col`, `narwhals.concat`, ...)
  - `LazyFrame.__native_namespace__`: return a native namespace object which must have a
    `from_dict` method
  - `Series.__narwhals_series__`: return an object which implements public methods
    from `Narwhals.Series`

  If your library doesn't distinguish between lazy and eager, then it's OK for your dataframe
  object to implement both `__narwhals_dataframe__` and `__narwhals_lazyframe__`. In fact,
  that's currently what `narwhals._pandas_like.dataframe.PandasLikeDataFrame` does. So, if you're stuck,
  take a look at the source code to see how it's done!

Note that this "extension" mechanism is still experimental. If anything is not clear, or
doesn't work, please do raise an issue or contact us on Discord (see the link on the README).
