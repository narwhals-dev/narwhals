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

## IO functions: the namespace contract

The Narwhals IO functions (`read_csv`, `scan_csv`, `read_parquet`, `scan_parquet`)
dispatch to same-named methods on the compliant namespace. This is a single mechanism,
shared by built-in backends and extensions alike: to support these functions, a
compliant namespace implements (a subset of):

- `read_csv(source, *, separator=",", **kwargs)`, returning a compliant DataFrame.
- `read_parquet(source, **kwargs)`, returning a compliant DataFrame.
- `scan_csv(source, *, separator=",", **kwargs)`, returning a compliant frame.
- `scan_parquet(source, **kwargs)`, returning a compliant frame.

In all cases:

- `source` is a plain string: Narwhals normalizes `Path` and path-like inputs before
  dispatching to the namespace.
- `kwargs` are forwarded to the native reader, and it is the namespace's responsibility
  to translate `separator` into whatever its native CSV reader expects (and to raise if
  the two conflict).
- `read_*` methods are eager-only, so they are only ever called on namespaces of eager
  backends. `scan_*` methods are called for any backend: lazy namespaces return a
  compliant LazyFrame, while eager ones may simply read eagerly. Namespaces complying
  with the `EagerNamespace` protocol only need to implement `read_csv` and
  `read_parquet`, as they inherit `scan_*` default implementations which fall back to
  the corresponding `read_*` method.

## Can I see an example?

Yes! For a reference plugin, please check out [narwhals-daft](https://github.com/narwhals-dev/narwhals-daft).
