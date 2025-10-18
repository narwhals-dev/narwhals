# Extending Narwhals

!!! warning

    The extension mechanism in Narwhals is experimental and under development.

If you want your own library to be recognised too, you're welcome open a PR (with tests)!.
Alternatively, if you can't do that (for example, if you library is closed-source), see
the next sections for what else you can do.

## Creating an Extension

We love open source, but we're not "open source absolutists". If you're unable to open
source your library, then this is how you can make your library compatible with Narwhals.

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

## Creating a Plugin

Another option is to write a plugin. Narwhals itself has the necessary utilities to detect and handle 
plugins. For this integration to work, any plugin architecture must contain the following:

  1. an entrypoint defined in a `pyproject.toml` file:

    ```
    [project.entry-points.'narwhals.plugins']
    narwhals-<library name> = 'narwhals_<library name>'
    ```
    The first line needs to be the same for all plugins, whereas the second is to be adapted to the 
    library name.

  2. a top-level `__init__.py` file containing the following: 
  
    - `is_native` and `__narwhals_namespace__` functions
    - a string constant `NATIVE_PACKAGE` which holds the name of the library for which the plugin is made

    `is_native` must receive a native object and return a boolean indicating whether the native object is 
    a dataframe of the plugin library.

    `__narwhals_namespace__` takes the Narwhals version and returns a compliant namespace for the library,
    i.e. one that complies with the CompliantNamespace protocol. This protocol specifies a `from_native` 
    function, whose input parameter is the Narwhals version and which returns a compliant Narwhals LazyFrame
    which wraps the native dataframe. 

If you want to see an example of a plugin, we have implemented a bare-bones version for the `daft` library
that allows users to pass daft dataframes to Narwhals: 
[narwhals-daft](https://github.com/MarcoGorelli/narwhals-daft). 
