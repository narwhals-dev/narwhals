# Extensions and Plugins

!!! warning

    The extension mechanism in Narwhals is experimental and under development.
    If anything is not clear, or doesn't work, please do raise an issue or
    contact us on Discord (see the link on the README).
  
If you would like to make a new library Narwhals-compatible, then there are
three ways to go about it:

- Open a PR to Narwhals. At this point, however, the bar for including new libraries
  is very high, so please discuss it with us beforehand.
- Make that library Narwhals-compliant.
- Make a plugin for that library which makes it Narwhals-compliant.

## Making a library Narwhals-compliant

It's possible to integrate your library with Narwhals with zero code changes
in Narwhals. To do this, you'll need to make sure that your library complies with
the Narwhals protocols found in `narwhals/compliant.py`.

For example, you'll need:

- A `LazyFrame` class which implements `__narwhals_lazyframe__`.
- A `Expr` class which implements `broadcast`.
- A `Namespace` class which implements `is_native`.
- ...

Full details can be found by inspecting the protocols in `narwhals/compliant.py`.

## Creating a Plugin

If it's not possible to add extra functions like `__narwhals_namespace__` and others to a dataframe object
itself, then another option is to write a plugin. Narwhals itself has the necessary utilities to detect and
handle plugins. For this integration to work, any plugin architecture must contain the following:

  1. an entrypoint defined in a `pyproject.toml` file:

    ```
    [project.entry-points.'narwhals.plugins']
    narwhals-<library name> = 'narwhals_<library name>'
    ```
    The section name needs to be the same for all plugins; inside it, plugin creators can replace their
    own library name, for example `narwhals-grizzlies = 'narwhals_grizzlies'`

  2. a top-level `__init__.py` file containing the following:
  
    - `is_native` and `__narwhals_namespace__` functions.
    - a string constant `NATIVE_PACKAGE` which holds the name of the library for which the plugin is made.

    `is_native` accepts a native object and returns a boolean indicating whether the native object is 
    a dataframe of the library the plugin was written for.

    `__narwhals_namespace__` takes the Narwhals version and returns a compliant namespace for the library,
    i.e. one that complies with the CompliantNamespace protocol. This protocol specifies a `from_native` 
    function, whose input parameter is the Narwhals version and which returns a compliant Narwhals LazyFrame
    which wraps the native dataframe. 
  
    Take a look at the `Plugin` protocol in `narwhals/plugins.py` for the
    signatures.

### Supporting `backend=...` in Narwhals functions

Functions and constructors which accept a `backend` argument can also dispatch to a
plugin. Users can pass:

- the plugin's entry point name (e.g. `backend="narwhals-grizzlies"`),
- the plugin's module name (e.g. `backend="narwhals_grizzlies"`),
- or the plugin's module itself (e.g. `backend=narwhals_grizzlies`).

There are two mechanisms, depending on the function:

1. **IO functions** (`read_csv`, `scan_csv`, `read_parquet`, `scan_parquet`): the plugin
   should expose a same-named function at the top level of its namespace which returns a
   **native** object (which is then passed through `narwhals.from_native`):

    - `read_csv(source, separator=",", **kwargs)`, `scan_csv(source, separator=",", **kwargs)`
    - `read_parquet(source, **kwargs)`, `scan_parquet(source, **kwargs)`

    If the required hook is missing, an informative `AttributeError` is raised.

2. **Eager constructors** (`from_dict`, `from_dicts`, `from_numpy`, `from_arrow`,
   `new_series`, as well as the `DataFrame.from_*` and `Series.from_*` classmethods):
   these are eager-only, and go through the compliant namespace returned by the plugin's
   `__narwhals_namespace__`. If that namespace implements the `EagerNamespace` protocol
   (in particular the `_dataframe` and `_series` properties, and the `from_dict`,
   `from_dicts`, `from_numpy`, `from_arrow` and `from_iterable` constructors on the
   respective compliant classes), these functions work with no extra plugin code.
   Lazy-only plugins get an informative `ValueError` instead.

Methods which internally construct Series (for example `Series.scatter`, or
`DataFrame.filter` with a list of booleans) use the compliant namespace of the object
they are called on, so they also work for eager plugins.

## Can I see an example?

Yes! For a reference plugin, please check out [narwhals-daft](https://github.com/narwhals-dev/narwhals-daft).
