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
from typing import Any, Final, NamedTuple

import polars as pl
from jinja2 import Template

from narwhals._utils import not_implemented

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


MODULES_CONFIG = {
    "dataframe": ["DataFrame", "LazyFrame"],
    "series": ["Series"],
    "expr": ["Expr"],
    "expr_dt": ["ExprDateTimeNamespace"],
    "expr_cat": ["ExprCatNamespace"],
    "expr_str": ["ExprStringNamespace"],
    "expr_list": ["ExprListNamespace"],
    "expr_name": ["ExprNameNamespace"],
    "expr_struct": ["ExprStructNamespace"],
    "series_dt": ["SeriesDateTimeNamespace"],
    "series_cat": ["SeriesCatNamespace"],
    "series_str": ["SeriesStringNamespace"],
    "series_list": ["SeriesListNamespace"],
    "series_struct": ["SeriesStructNamespace"],
    "functions": ["functions"],
}
"""Mapping of narwhals top-level modules to their main classes."""

BACKENDS = (
    Backend(name="arrow", type_=BackendType.EAGER),
    Backend(name="dask", type_=BackendType.LAZY),
    Backend(name="duckdb", type_=BackendType.LAZY),
    Backend(name="ibis", type_=BackendType.LAZY),
    Backend(name="pandas-like", type_=BackendType.EAGER),
    Backend(name="spark-like", type_=BackendType.LAZY),
)

# Methods that are always implemented at the wrapper level
ALWAYS_IMPLEMENTED = {"pipe", "to_native"}

# Backends that reuse Series implementations for Expr (and subnamespaces)
SERIES_REUSING_BACKENDS = {"arrow", "pandas-like"}

# Constructor functions available for eager backends
EAGER_CONSTRUCTOR_METHODS = {"from_arrow", "from_dict", "from_dicts", "from_numpy"}


def _is_eager_allowed(backend: Backend) -> bool:
    """Check if a backend supports eager evaluation."""
    return backend.type_ in {BackendType.EAGER, BackendType.BOTH}


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
        # Skip non public methods
        if name.startswith("_"):
            continue

        try:
            attr = inspect.getattr_static(kls, name)

            # Check if it's a `not_implemented` marker. For properties, check if the `fget` is `not_implemented`
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
    # Special case: for functions, look for the Namespace class
    if module_name == "functions":
        module_name, target_class_name = "namespace", "Namespace"

    try:
        module = importlib.import_module(f"narwhals.{compliant_module}.{module_name}")
    except ModuleNotFoundError:
        return None

    # Look for classes that match the pattern
    # Backend classes usually have names like ArrowDataFrame, PandasLikeDataFrame, etc.
    for name, obj in inspect.getmembers(module, inspect.isclass):
        # Skip protocols and test classes
        if name.startswith(("Compliant", "DuckDBInterchange")):
            continue

        # Check if this class is defined in this module (not imported)
        if obj.__module__ != module.__name__:
            continue

        # Match pattern: class name should end with or contain the target class name
        if target_class_name in name:
            return obj

    return None


def _add_series_reusing_methods(
    methods: set[str], module_name: str, backend: Backend, target_class_name: str
) -> set[str]:
    """Add methods from Series implementations for arrow and pandas-like Expr classes."""
    if backend.name not in SERIES_REUSING_BACKENDS or not module_name.startswith("expr"):
        return methods

    # Map expr module to corresponding series module
    series_module = module_name.replace("expr", "series")
    series_class_name = target_class_name.replace("Expr", "Series")

    # Get methods from the corresponding Series class
    series_class = find_compliant_class(series_module, backend.module, series_class_name)

    if series_class is not None:
        series_methods = get_implemented_methods_from_class(series_class)
        methods.update(series_methods)
    return methods


def _add_string_namespace_methods(methods: set[str], module_name: str) -> set[str]:
    """Add head/tail methods for StringNamespace when slice is available."""
    if module_name in {"expr_str", "series_str"} and "slice" in methods:
        methods.update({"head", "tail"})
    return methods


def _add_functions_methods(methods: set[str], backend: Backend) -> set[str]:
    """Add composed and wrapper-level methods for the functions module."""
    # Get Expr methods to check for composed functions
    expr_methods = get_backend_methods("expr", backend, "Expr")

    # Aggregation functions: implemented via col + Expr.<method>
    if "col" in methods:
        for agg_method in ("min", "max", "mean", "median", "sum"):
            if agg_method in expr_methods:
                methods.add(agg_method)

    # format is implemented via concat_str
    if "concat_str" in methods:
        methods.add("format")

    # when is implemented via when_then in the namespace
    if "when_then" in methods:
        methods.add("when")

    # I/O functions: read_* are eager-only, scan_* are for all backends
    if _is_eager_allowed(backend):
        methods.update({"read_csv", "read_parquet"})
    methods.update({"scan_csv", "scan_parquet"})

    # Constructor functions (eager only)
    if _is_eager_allowed(backend):
        methods.update(EAGER_CONSTRUCTOR_METHODS | {"new_series"})
    return methods


def _add_series_methods(methods: set[str], backend: Backend) -> set[str]:
    """Add composed and wrapper-level methods for the Series class."""
    # is_close is composed from other operations
    methods.add("is_close")

    # hist is implemented via hist_from_bins and hist_from_bin_count
    if "hist_from_bins" in methods and "hist_from_bin_count" in methods:
        methods.add("hist")

    # Wrapper-level methods (eager only)
    if _is_eager_allowed(backend):
        methods.update({"from_iterable", "from_numpy", "shape"})

    # rename is implemented via alias
    if "alias" in methods:
        methods.add("rename")

    return methods


def _add_dataframe_methods(methods: set[str], backend: Backend) -> set[str]:
    """Add composed and wrapper-level methods for the DataFrame class."""
    # Constructor methods (eager only)
    if _is_eager_allowed(backend):
        methods.update(EAGER_CONSTRUCTOR_METHODS)

    # is_duplicated is implemented via is_unique
    if "is_unique" in methods:
        methods.add("is_duplicated")

    # null_count is implemented via Expr.null_count
    expr_methods = get_backend_methods("expr", backend, "Expr")
    if "null_count" in expr_methods:
        methods.add("null_count")

    # is_empty is implemented via len()
    methods.add("is_empty")
    return methods


def get_backend_methods(
    module_name: str, backend: Backend, target_class_name: str
) -> set[str]:
    """Get all implemented methods for a backend's implementation of a class."""
    # Special case: ExprNameNamespace is implemented via CompliantExprNameNamespace
    # at the compliant level, so all backends with Expr support have all its methods
    if module_name == "expr_name" and target_class_name == "ExprNameNamespace":
        return get_narwhals_methods(module_name, target_class_name)

    backend_class = find_compliant_class(module_name, backend.module, target_class_name)

    methods = set()
    if backend_class is not None:
        methods = get_implemented_methods_from_class(backend_class)

    # Add methods from Series implementations for Expr classes (arrow, pandas-like)
    methods = _add_series_reusing_methods(
        methods, module_name, backend, target_class_name
    )

    # Add head/tail for StringNamespace when slice is available
    methods = _add_string_namespace_methods(methods, module_name)

    # Add module-specific composed and wrapper-level methods
    if module_name == "functions":
        methods = _add_functions_methods(methods, backend)
    elif module_name == "series" and target_class_name == "Series":
        methods = _add_series_methods(methods, backend)
    elif module_name == "dataframe" and target_class_name == "DataFrame":
        methods = _add_dataframe_methods(methods, backend)
    elif module_name == "expr" and target_class_name == "Expr":
        methods.add("is_close")

    # Add always-implemented methods
    methods.update(ALWAYS_IMPLEMENTED)

    return methods


def get_narwhals_methods(module_name: str, class_name: str) -> set[str]:
    """Get all public methods from a narwhals top-level class.

    For the special case of 'functions', returns all public functions from the module.
    """
    try:
        module = importlib.import_module(f"narwhals.{module_name}")
    except (ModuleNotFoundError, AttributeError):
        return set()

    # Special case: for functions module, get all public functions
    if module_name == class_name == "functions":
        return _get_public_functions_from_module(module)

    kls = getattr(module, class_name, None)
    if kls is None:
        return set()

    return _get_public_methods_and_properties(kls)


def _get_public_functions_from_module(module: Any) -> set[str]:
    """Get all public functions from a module.

    For the functions module, we only include functions that are exported
    in narwhals.__init__.py to avoid including internal utilities.
    """
    # Get the list of functions exported from narwhals.__init__
    import narwhals

    all_exports = getattr(narwhals, "__all__", [])

    # Functions to exclude from the completeness table
    # show_versions is metadata-only, not a backend-specific function
    excluded_functions = {"show_versions"}

    # Get all functions from the module
    methods = set()
    for name in dir(module):
        if name.startswith("_") or name in excluded_functions:
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


def create_completeness_dataframe(module_name: str, class_name: str) -> pl.DataFrame:
    """Create a dataframe showing backend completeness for a specific class."""
    # Get narwhals methods
    nw_methods = get_narwhals_methods(module_name, class_name)

    if not nw_methods:
        return pl.DataFrame()

    data = [
        {"Backend": "narwhals", "Method": method, "Supported": True}
        for method in sorted(nw_methods)
    ]

    for backend in _get_relevant_backends(class_name):
        backend_methods = get_backend_methods(module_name, backend, class_name)
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
    if df.is_empty():
        print(f"Warning: No data for {title}")
        return

    # Pivot to create the comparison table
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

    # Reorder columns: Method, then alphabetically sorted backends, then polars
    backend_cols = [c for c in table_df.columns if c != "Method"]

    # Add polars column (always supported)
    if "narwhals" in backend_cols:
        table_df = table_df.drop("narwhals")
        backend_cols.remove("narwhals")

    # Add polars as always supported
    table_df = table_df.with_columns(polars=pl.lit(":white_check_mark:"))

    # Reorder columns
    final_cols = ["Method", *sorted(backend_cols), "polars"]
    table_df = table_df.select(final_cols)

    # Format as markdown table
    with pl.Config(
        tbl_formatting="ASCII_MARKDOWN",
        tbl_hide_column_data_types=True,
        tbl_hide_dataframe_shape=True,
        set_tbl_rows=table_df.shape[0],
        set_tbl_width_chars=1_000,
    ):
        table = str(table_df)

    # Write to file
    with TEMPLATE_PATH.open(mode="r") as stream:
        content = Template(stream.read()).render({"backend_table": table, "title": title})

    output_path = DESTINATION_PATH / f"{output_filename}.md"
    with output_path.open(mode="w") as f:
        f.write(content)

    print(f"Generated: {output_path}")


def generate_completeness_tables() -> None:
    """Generate all backend completeness tables."""
    for module_name, class_names in MODULES_CONFIG.items():
        print(f"\nProcessing module: {module_name}")

        for class_name in class_names:
            print(f"  Processing class: {class_name}")

            df = create_completeness_dataframe(module_name, class_name)

            if not df.is_empty():
                # Determine title and filename
                if class_name in {"DataFrame", "LazyFrame"}:
                    title, filename = class_name, class_name.lower()
                elif module_name == "functions":
                    title = filename = "functions"
                else:
                    title = f"{module_name}.{class_name}".replace("_", ".")
                    filename = module_name

                render_table_and_write_to_output(df, title, filename)


generate_completeness_tables()
