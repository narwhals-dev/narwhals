# List of supported libraries (and how to add yours!)

Currently, Narwhals supports the following libraries as inputs:

- pandas
- Polars
- cuDF
- Modin

If you want your own library to be recognised too, you're welcome open a PR (with tests)!
Alternatively, if you can't do that (for example, if you library is closed-source), see
the next section for what else you can do.

## Extending Narwhals

We love open source, but we're not "open source absolutists". If you're unable to open
source you library, then this is how you can make your library compatible with Narwhals.

Make sure that, in addition to the public Narwhals API, you also define:

  - `DataFrame.__narwhals_dataframe__`: return an object which implements public methods
    from `Narwhals.DataFrame`
  - `DataFrame.__narwhals_namespace__`: return an object which implements public top-level
    functions from `narwhals` (e.g. `narwhals.col`, `narwhals.concat`, ...)
  - `LazyFrame.__narwhals_lazyframe__`: return an object which implements public methods
    from `Narwhals.LazyFrame`
  - `LazyFrame.__narwhals_namespace__`: return an object which implements public top-level
    functions from `narwhals` (e.g. `narwhals.col`, `narwhals.concat`, ...)
  - `Series.__narwhals_series__`: return an object which implements public methods
    from `Narwhals.Series`
  - `Series.__narwhals_namespace__`: return an object which implements public top-level
    functions from `narwhals` (e.g. `narwhals.col`, `narwhals.concat`, ...)

  If your library doesn't distinguish between lazy and eager, then it's OK for your dataframe
  object to implement both `__narwhals_dataframe__` and `__narwhals_lazyframe__`. In fact,
  that's currently what `narwhals._pandas_like.dataframe.PandasDataFrame` does. So, if you're stuck,
  take a look at the source code to see how it's done!

Note that the "extension" mechanism is still experimental. If anything is not clear, or
doesn't work, please do raise an issue or contact us on Discord (see the link on the README).
