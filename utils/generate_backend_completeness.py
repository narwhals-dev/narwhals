"""Generate backend API completeness tables.

Analyzes the narwhals codebase to create tables showing which methods are implemented
by each backend (arrow, dask, duckdb, ibis, pandas-like, spark-like).

For each class (DataFrame, LazyFrame, Series, Expr, etc.), the script:
- Discovers all public methods
- Checks backend implementations, handling `not_implemented` markers
- Accounts for lazy vs eager backends and Series-reusing patterns in Expr classes
- Generates markdown tables in docs/api-completeness/ with ✓/✗ indicators
"""

from __future__ import annotations

import importlib
import inspect
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, NamedTuple

import polars as pl
from jinja2 import Template

from narwhals._utils import not_implemented

if TYPE_CHECKING:
    from types import ModuleType

TEMPLATE_PATH: Final[Path] = Path("utils") / "api-completeness.md.jinja"
DESTINATION_PATH: Final[Path] = Path("docs") / "api-completeness"


class BackendType(Enum):
    LAZY = auto()
    EAGER = auto()
    BOTH = auto()


class Backend(NamedTuple):
    name: str
    type_: BackendType

    @property
    def module(self) -> str:
        return f"_{self.name.replace('-', '_')}"

    @property
    def is_eager_allowed(self) -> bool:
        """Check if a backend supports eager evaluation."""
        return self.type_ in {BackendType.EAGER, BackendType.BOTH}


CLASS_NAME_TO_MODULE_NAME = {
    "DataFrame": "dataframe",
    "LazyFrame": "dataframe",
    "Series": "series",
    "Expr": "expr",
    "ExprDateTimeNamespace": "expr_dt",
    "ExprCatNamespace": "expr_cat",
    "ExprStringNamespace": "expr_str",
    "ExprListNamespace": "expr_list",
    "ExprNameNamespace": "expr_name",
    "ExprStructNamespace": "expr_struct",
    "SeriesDateTimeNamespace": "series_dt",
    "SeriesCatNamespace": "series_cat",
    "SeriesStringNamespace": "series_str",
    "SeriesListNamespace": "series_list",
    "SeriesStructNamespace": "series_struct",
    "functions": "functions",
}
"""Mapping of narwhals public classes and functions to the modules where they are defined."""

BACKENDS = (
    Backend(name="arrow", type_=BackendType.EAGER),
    Backend(name="dask", type_=BackendType.LAZY),
    Backend(name="duckdb", type_=BackendType.LAZY),
    Backend(name="ibis", type_=BackendType.LAZY),
    Backend(name="pandas-like", type_=BackendType.EAGER),
    Backend(name="spark-like", type_=BackendType.LAZY),
)

ALWAYS_IMPLEMENTED = {"pipe", "to_native"}
"""Methods that are always implemented at the wrapper level"""

SERIES_REUSING_BACKENDS = {"arrow", "pandas-like"}
"""Backends that reuse Series implementations for Expr (and subnamespaces)"""

DATAFRAME_CONSTRUCTOR_METHODS = {"from_arrow", "from_dict", "from_dicts", "from_numpy"}
"""Constructor functions available for eager backends"""

DEPRECATED_METHODS = {
    "Expr": {
        "arg_max",
        "arg_min",
        "arg_true",
        "gather_every",
        "head",
        "sample",
        "sort",
        "tail",
    },
    "LazyFrame": {"gather_every", "tail"},
}
"""Deprecated methods to exclude from completeness tables for a given class"""

EXCLUDED_FUNCTIONS = {"show_versions"}
"""Functions to exclude, because not backend specific"""


def _get_public_methods_and_properties(obj: type[Any]) -> set[str]:
    """Get all public method and property names from a class."""
    methods = set()
    for name in dir(obj):
        if name.startswith("_"):
            continue
        attr = getattr(obj, name, None)
        if callable(attr) or isinstance(attr, property):
            methods.add(name)
    return methods


def get_implemented_methods_from_class(kls: type[Any]) -> set[str]:
    """Get all public methods from a class that are actually implemented.

    Walks through the MRO and checks for not_implemented markers.
    """
    implemented = set()

    for name in dir(kls):
        if name.startswith("_"):
            continue

        try:
            attr = inspect.getattr_static(kls, name)

            if isinstance(attr, not_implemented) or (
                isinstance(attr, property) and isinstance(attr.fget, not_implemented)
            ):
                continue

            if callable(attr) or isinstance(attr, property):
                implemented.add(name)

        except AttributeError:
            continue

    return implemented


def find_compliant_class(
    module_name: str, compliant_module: str, target_class_name: str
) -> type[Any] | None:
    """Find the compliant implementation class for a given narwhals class.

    Arguments:
        module_name: The narwhals module name (e.g., "dataframe")
        compliant_module: The compliant module path (e.g., "_arrow")
        target_class_name: The class name to find (e.g., "DataFrame")

    Returns:
        The compliant class if found, None otherwise.
    """
    if module_name == "functions":
        # Special case: for functions, look for the Namespace class
        module_name, target_class_name = "namespace", "Namespace"

    try:
        module = importlib.import_module(f"narwhals.{compliant_module}.{module_name}")
    except ModuleNotFoundError:
        return None

    for cls_name, obj in inspect.getmembers(module, inspect.isclass):
        if (
            cls_name.startswith("Compliant")
            or "Interchange" in cls_name
            or obj.__module__ != module.__name__
        ):
            # Skip protocols, Interchange classes and if classes not defined in the module
            continue

        if cls_name.endswith(target_class_name):
            # Compliant classes follow the pattern <Backend><cls_name> e.g. ArrowDataFrame, DuckDBExpr
            return obj

    return None


def _add_series_reusing_methods(
    methods: set[str], module_name: str, backend: Backend, target_class_name: str
) -> set[str]:
    """Add methods from Series implementations for arrow and pandas-like Expr classes."""
    series_module = module_name.replace("expr", "series")
    series_class_name = target_class_name.replace("Expr", "Series")

    compliant_series_class = find_compliant_class(
        series_module, backend.module, series_class_name
    )

    if compliant_series_class is not None:
        series_methods = get_implemented_methods_from_class(compliant_series_class)
        methods.update(series_methods)
    return methods


def _add_string_namespace_methods(methods: set[str], module_name: str) -> set[str]:
    """Add head/tail methods for StringNamespace when slice is available."""
    if module_name in {"expr_str", "series_str"} and "slice" in methods:
        methods.update({"head", "tail"})
    return methods


def _add_functions_methods(methods: set[str], backend: Backend) -> set[str]:
    """Add composed and wrapper-level methods for the functions module.

    * `nw.{max, mean, median, min, sum}` are implemented via `nw.col + Expr.<method>`
    * `nw.format` is implemented via `nw.concat_str`
    * `nw.when` is implemented via `CompliantNamespace.when_then`
    * I/O functions: `read_*` are eager-only, `scan_*` are for all backends, implemented at top level
    * `nw.{from_arrow, from_dict, from_dicts, from_numpy, new_series}` are implemented at top level for eager only
    """
    expr_methods = get_backend_methods(backend, "expr", "Expr")

    if "col" in methods:
        for agg_method in ("min", "max", "mean", "median", "sum"):
            if agg_method in expr_methods:
                methods.add(agg_method)

    if "concat_str" in methods:
        methods.add("format")
    if "when_then" in methods:
        methods.add("when")

    if backend.is_eager_allowed:
        methods.update(
            {"read_csv", "read_parquet", "new_series"} | DATAFRAME_CONSTRUCTOR_METHODS
        )

    methods.update({"scan_csv", "scan_parquet"})
    return methods


def _add_series_methods(methods: set[str], backend: Backend) -> set[str]:
    """Add composed and wrapper-level methods for the Series class.

    * `is_close` is implemented at the narwhals level via other Series methods
    * `hist` is implemented via `hist_from_bins` and `hist_from_bin_count`
    * `Series` class constructors
    * `rename` is an alias for `alias`
    """
    methods.add("is_close")

    if "hist_from_bins" in methods and "hist_from_bin_count" in methods:
        methods.add("hist")

    if "alias" in methods:
        methods.add("rename")

    if backend.is_eager_allowed:
        methods.update({"from_iterable", "from_numpy", "shape"})

    return methods


def _add_dataframe_methods(methods: set[str], backend: Backend) -> set[str]:
    """Add composed and wrapper-level methods for the DataFrame class.

    * Eager: `DataFrame` class constructors and `is_empty`
    * `is_duplicated` is implemented via `is_unique`
    * `null_count` is implemented via `Expr.null_count`
    """
    if backend.is_eager_allowed:
        methods.update(DATAFRAME_CONSTRUCTOR_METHODS | {"is_empty"})

    if "is_unique" in methods:
        methods.add("is_duplicated")

    expr_methods = get_backend_methods(backend, "expr", "Expr")
    if "null_count" in expr_methods:
        methods.add("null_count")

    return methods


def get_backend_methods(
    backend: Backend, module_name: str, target_class_name: str
) -> set[str]:
    """Get all implemented methods for a backend's implementation of a class.

    'Special' cases:

    * `ExprNameNamespace` is implemented via `CompliantExprNameNamespace` at the compliant level,
        so all backends with `Expr` support have all its methods
    * Arrow and Pandas-like `Expr` methods reuse `Series` implementation
    * `{Expr, Series}.str.{head, tail}` are implemented via `{Expr, Series}.str.slice`
    *
    """
    if module_name == "expr_name" and target_class_name == "ExprNameNamespace":
        return get_narwhals_methods(module_name, target_class_name)

    compliant_class = find_compliant_class(module_name, backend.module, target_class_name)

    methods = set(ALWAYS_IMPLEMENTED)
    if compliant_class is not None:
        methods.update(get_implemented_methods_from_class(compliant_class))

    if backend.name in SERIES_REUSING_BACKENDS and module_name.startswith("expr"):
        methods = _add_series_reusing_methods(
            methods, module_name, backend, target_class_name
        )

    methods = _add_string_namespace_methods(methods, module_name)

    if module_name == "functions":
        methods = _add_functions_methods(methods, backend)
    elif target_class_name == "DataFrame":
        methods = _add_dataframe_methods(methods, backend)
    elif target_class_name == "Series":
        methods = _add_series_methods(methods, backend)
    elif target_class_name == "Expr":
        methods.add("is_close")

    return methods


def get_narwhals_methods(module_name: str, class_name: str) -> set[str]:
    """Get all public methods from a narwhals top-level class.

    For the special case of 'functions', returns all public functions from the module.
    """
    try:
        module = importlib.import_module(f"narwhals.{module_name}")
    except (ModuleNotFoundError, AttributeError):
        return set()

    if module_name == class_name == "functions":
        return _get_public_functions_from_module(module)

    kls = getattr(module, class_name, None)
    if kls is None:
        return set()

    return _get_public_methods_and_properties(kls)


def _get_public_functions_from_module(module: ModuleType) -> set[str]:
    """Get all public functions from a module.

    For the functions module, we only include functions that are exported
    in narwhals.__init__.py to avoid including internal utilities.
    """
    import narwhals

    all_exports = getattr(narwhals, "__all__", [])
    methods = set()

    for name in dir(module):
        if name.startswith("_") or name in EXCLUDED_FUNCTIONS:
            continue
        attr = getattr(module, name)
        if not (callable(attr) and inspect.isfunction(attr)):
            continue

        # Check if this function (or its alias) is exported in __all__
        # Handle aliases like all_ -> all, len_ -> len
        if (exported_name := name.rstrip("_")) in all_exports:
            methods.add(exported_name)
    return methods


def _get_relevant_backends(class_name: str) -> tuple[Backend, ...]:
    """Get backends relevant for a specific class.

    - LazyFrame: only lazy backends
    - DataFrame, Series, Series*: only eager backends
    - Expr, Expr*, functions: all backends
    """
    if class_name == "LazyFrame":
        return tuple(b for b in BACKENDS if b.type_ == BackendType.LAZY)
    if class_name == "DataFrame" or class_name.startswith("Series"):
        return tuple(b for b in BACKENDS if b.type_ == BackendType.EAGER)
    return BACKENDS


def _filter_deprecated_methods(methods: set[str], class_name: str) -> set[str]:
    """Remove deprecated methods from the set of methods.

    Arguments:
        methods: Set of method names to filter
        class_name: The class name (e.g., "Expr", "LazyFrame")

    Returns:
        Filtered set of methods with deprecated ones removed
    """
    deprecated = DEPRECATED_METHODS.get(class_name, set())
    return methods - deprecated


def create_completeness_dataframe(module_name: str, class_name: str) -> pl.DataFrame:
    """Create a dataframe showing backend completeness for a specific class."""
    nw_methods = get_narwhals_methods(module_name, class_name)
    nw_methods = _filter_deprecated_methods(nw_methods, class_name)

    if not nw_methods:
        msg = f"Could not generate completeness dataframe for {module_name}.{class_name}"
        raise AssertionError(msg)

    data = [
        {"Backend": "polars", "Method": method, "Supported": True}
        for method in sorted(nw_methods)
    ]

    for backend in _get_relevant_backends(class_name):
        backend_methods = get_backend_methods(backend, module_name, class_name)
        data.extend(
            {
                "Backend": backend.name,
                "Method": method,
                "Supported": method in backend_methods,
            }
            for method in sorted(nw_methods)
        )
    return pl.DataFrame(data)


def render_table_and_write_to_output(
    df: pl.DataFrame, title: str, output_filename: str
) -> None:
    """Render a markdown table and write it to a file."""
    table_df = (
        df.with_columns(
            pl.when(pl.col("Supported"))
            .then(pl.lit(":white_check_mark:"))
            .otherwise(pl.lit(":x:"))
            .alias("mark")
        )
        .pivot(on="Backend", values="mark", index="Method", aggregate_function="first")
        .sort("Method")
    )

    # Reorder columns: Method, then alphabetically sorted backends
    backend_cols = [c for c in table_df.columns if c != "Method"]
    final_cols = ["Method", *sorted(backend_cols)]
    table_df = table_df.select(final_cols)

    with pl.Config(
        tbl_formatting="ASCII_MARKDOWN",
        tbl_hide_column_data_types=True,
        tbl_hide_dataframe_shape=True,
        set_tbl_rows=table_df.shape[0],
        set_tbl_width_chars=1_000,
    ):
        table = str(table_df)

    with TEMPLATE_PATH.open(mode="r") as stream:
        content = Template(stream.read()).render({"backend_table": table, "title": title})

    output_path = DESTINATION_PATH / f"{output_filename}.md"
    with output_path.open(mode="w") as f:
        f.write(content)

    print(f"Generated: {output_path}")


def create_title_and_filename(module_name: str, class_name: str) -> tuple[str, str]:
    """Create markdown filename and page title from class and module name."""
    if module_name == "dataframe":
        filename, title = class_name.lower(), class_name
    elif module_name == "functions":
        filename = title = "functions"
    else:
        filename = module_name
        title = f"{module_name}.{class_name}".replace("_", ".")

    return filename, title


def generate_completeness_tables() -> None:
    """Generate all backend completeness tables."""
    for class_name, module_name in CLASS_NAME_TO_MODULE_NAME.items():
        completeness_frame = create_completeness_dataframe(module_name, class_name)
        filename, title = create_title_and_filename(module_name, class_name)

        render_table_and_write_to_output(completeness_frame, title, filename)


generate_completeness_tables()
