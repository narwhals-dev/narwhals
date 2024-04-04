# How Narwhals can support your dataframe as well!

Currently, Narwhals recognises the following libraries as inputs:

- pandas
- Polars
- cuDF
- Modin

If you want your own library to be recognised too, you can either open a PR (with tests) or
you can make sure that, in addition to the public Narwhals API, you also define:

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
  that's currently what `narwhals._pandas_like.dataframe.PandasDataFrame` does!
