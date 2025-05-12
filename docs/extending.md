# Extending Narwhals

!!! warning

    The extension mechanism in Narwhals is experimental and under development.

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
