"""Generate backend API completeness tables.

This script analyzes the narwhals codebase to create comprehensive tables showing which
methods are implemented by each backend. It performs the following:

1. For each narwhals top-level class (DataFrame, LazyFrame, Series, Expr, etc.):
   - Discovers all public methods and properties
   - Checks each backend's implementation for those methods
   - Properly handles `not_implemented` markers in the MRO
   - Accounts for lazy vs eager backend differences

2. Key features:
   - Walks through the full Method Resolution Order (MRO) to find implementations
   - Detects `not_implemented` markers on both methods and properties
   - Maps DataFrame to LazyFrame for lazy-only backends (dask, duckdb, spark-like)
   - For arrow and pandas-like backends, detects Expr methods that reuse Series
     implementations (via the `_reuse_series` pattern in EagerExpr)
   - Generates markdown tables with visual indicators (✓ or ✗)

3. Series-reusing pattern:
   - Arrow and pandas-like backends inherit from EagerExpr, which provides
     `_reuse_series` methods to delegate Expr implementations to Series
   - When processing expr/expr_* modules, the script automatically checks the
     corresponding series/series_* modules to find inherited implementations
   - This ensures that methods like `filter`, `contains`, `len_chars`, etc. are
     correctly marked as supported when they exist in the Series implementation

4. Output:
   - Creates separate markdown files for each class in docs/api-completeness/
   - Tables show: Method name | backend support | polars (always ✓)
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
    module: str
    type_: BackendType


# Mapping of narwhals top-level modules to their main classes
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
}

BACKENDS = [
    Backend(name="arrow", module="_arrow", type_=BackendType.EAGER),
    Backend(name="dask", module="_dask", type_=BackendType.LAZY),
    Backend(name="duckdb", module="_duckdb", type_=BackendType.LAZY),
    Backend(name="pandas-like", module="_pandas_like", type_=BackendType.EAGER),
    Backend(name="spark-like", module="_spark_like", type_=BackendType.LAZY),
]

# Methods that are always implemented at the wrapper level
ALWAYS_IMPLEMENTED = {"pipe", "to_native"}

# Backends that reuse Series implementations for Expr (and subnamespaces)
SERIES_REUSING_BACKENDS = {"arrow", "pandas-like"}


def inherits_from_eager_expr(kls: type[Any]) -> bool:
    """Check if a class inherits from EagerExpr (uses _reuse_series pattern)."""
    return any(base.__name__ == "EagerExpr" for base in inspect.getmro(kls))


def get_implemented_methods_from_class(kls: type[Any]) -> set[str]:
    """Get all public methods from a class that are actually implemented.

    Walks through the MRO and checks for not_implemented markers.
    """
    implemented = set()

    for name in dir(kls):
        if name.startswith("_"):
            continue

        try:
            # Get the attribute from the class (not instance)
            attr = inspect.getattr_static(kls, name)

            # Check if it's a not_implemented marker
            if isinstance(attr, not_implemented):
                continue

            # For properties, check if the fget is not_implemented
            if isinstance(attr, property) and isinstance(attr.fget, not_implemented):
                continue

            # If we get here, it's implemented
            if callable(attr) or isinstance(attr, property):
                implemented.add(name)
        except AttributeError:
            # If we can't get the attribute, skip it
            continue

    return implemented


def find_backend_class(
    module_name: str,
    backend_module: str,
    target_class_name: str,
    backend_type: BackendType,
) -> type[Any] | None:
    """Find the backend implementation class for a given narwhals class.

    Args:
        module_name: The narwhals module name (e.g., "dataframe")
        backend_module: The backend module path (e.g., "_arrow")
        target_class_name: The class name to find (e.g., "DataFrame")
        backend_type: Whether the backend is lazy or eager

    Returns:
        The backend class if found, None otherwise.
    """
    try:
        module = importlib.import_module(f"narwhals.{backend_module}.{module_name}")
    except ModuleNotFoundError:
        return None

    # For lazy backends processing DataFrame, look for LazyFrame instead
    search_name = target_class_name
    if target_class_name == "DataFrame" and backend_type == BackendType.LAZY:
        search_name = "LazyFrame"

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
        if search_name in name:
            return obj

    return None


def get_backend_methods(
    module_name: str, backend: Backend, target_class_name: str
) -> set[str]:
    """Get all implemented methods for a backend's implementation of a class."""
    # Special case: ExprNameNamespace is implemented via CompliantExprNameNamespace
    # at the compliant level, so all backends with Expr support have all its methods
    if module_name == "expr_name" and target_class_name == "ExprNameNamespace":
        # Check if this backend has Expr support
        expr_class = find_backend_class("expr", backend.module, "Expr", backend.type_)
        if expr_class is not None:
            # If backend has Expr, it has all name namespace methods
            nw_methods = get_narwhals_methods(module_name, target_class_name)
            return nw_methods | ALWAYS_IMPLEMENTED
        return set()

    backend_class = find_backend_class(
        module_name, backend.module, target_class_name, backend.type_
    )

    methods = set()
    if backend_class is not None:
        methods = get_implemented_methods_from_class(backend_class)

    # For arrow and pandas-like backends, Expr classes reuse Series implementations
    # This applies to:
    # - Expr -> Series (main expression class)
    # - expr_str -> series_str (ExprStringNamespace -> SeriesStringNamespace)
    # - expr_list -> series_list (ExprListNamespace -> SeriesListNamespace)
    # - expr_dt -> series_dt (ExprDateTimeNamespace -> SeriesDateTimeNamespace)
    # - expr_cat -> series_cat (ExprCatNamespace -> SeriesCatNamespace)
    # - expr_struct -> series_struct (ExprStructNamespace -> SeriesStructNamespace)
    if backend.name in SERIES_REUSING_BACKENDS and module_name.startswith("expr"):
        # Map expr module to corresponding series module
        series_module = module_name.replace("expr", "series")
        series_class_name = target_class_name.replace("Expr", "Series")

        # Get methods from the corresponding Series class
        series_class = find_backend_class(
            series_module, backend.module, series_class_name, backend.type_
        )

        if series_class is not None:
            series_methods = get_implemented_methods_from_class(series_class)
            # Add all series methods to expr methods (union)
            methods.update(series_methods)

    # Special case: StringNamespace.head and StringNamespace.tail are implemented via slice
    # This applies to both ExprStringNamespace and SeriesStringNamespace
    if module_name in {"expr_str", "series_str"} and "slice" in methods:
        methods.update({"head", "tail"})

    # Add always-implemented methods
    methods.update(ALWAYS_IMPLEMENTED)

    return methods


def get_narwhals_methods(module_name: str, class_name: str) -> set[str]:
    """Get all public methods from a narwhals top-level class."""
    try:
        module = importlib.import_module(f"narwhals.{module_name}")
        kls = getattr(module, class_name, None)

        if kls is None:
            return set()

        # Get all public methods/properties
        methods = set()
        for name in dir(kls):
            if not name.startswith("_"):
                attr = getattr(kls, name)
                if callable(attr) or isinstance(attr, property):
                    methods.add(name)
    except (ModuleNotFoundError, AttributeError):
        return set()
    else:
        return methods


def create_completeness_dataframe(module_name: str, class_name: str) -> pl.DataFrame:
    """Create a dataframe showing backend completeness for a specific class."""
    # Get narwhals methods
    nw_methods = get_narwhals_methods(module_name, class_name)

    if not nw_methods:
        return pl.DataFrame()

    # Collect data for all backends
    data = [
        {"Backend": "narwhals", "Method": method, "Supported": True}
        for method in sorted(nw_methods)
    ]

    # Add backend rows
    for backend in BACKENDS:
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
                    title = class_name
                    filename = class_name.lower()
                else:
                    title = f"{module_name}.{class_name}".replace("_", ".")
                    filename = module_name

                render_table_and_write_to_output(df, title, filename)


if __name__ == "__main__":
    generate_completeness_tables()
