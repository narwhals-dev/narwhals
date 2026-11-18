"""A pure-Python, eager dataframe library whose native frame is a `dict[str, list]`.

Use it standalone, with Narwhals' API:

    import pizzlies as pz

    df = pz.from_dict({"a": [1, 2, 3]})
    df.with_columns(b=pz.col("a") * 2)

or as a Narwhals plugin, where plain dicts become valid native frames:

    import narwhals as nw

    nw.from_native({"a": [1, 2, 3]}, eager_only=True)

Parquet files are read by a small pure-Python reader, which supports flat columns and the
common encodings and compression codecs (zstd needs Python 3.14+).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import narwhals as nw
from narwhals import (
    Array,
    Binary,
    Boolean,
    Categorical,
    DataFrame,
    Date,
    Datetime,
    Decimal,
    Duration,
    Enum,
    Expr,
    Field,
    Float16,
    Float32,
    Float64,
    Implementation,
    Int8,
    Int16,
    Int32,
    Int64,
    Int128,
    LazyFrame,
    List,
    Map,
    Object,
    Schema,
    Series,
    String,
    Struct,
    Time,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    UInt128,
    Unknown,
    all,
    all_horizontal,
    any_horizontal,
    coalesce,
    col,
    concat,
    concat_str,
    corr,
    cov,
    dependencies,
    dtypes,
    exceptions,
    exclude,
    format,
    from_native,
    generate_temporary_column_name,
    get_native_namespace,
    is_ordered_categorical,
    len,
    list,
    lit,
    max,
    max_horizontal,
    maybe_align_index,
    maybe_convert_dtypes,
    maybe_get_index,
    maybe_reset_index,
    maybe_set_index,
    mean,
    mean_horizontal,
    median,
    min,
    min_horizontal,
    narwhalify,
    nth,
    selectors,
    show_versions,
    struct,
    sum,
    sum_horizontal,
    to_native,
    to_py_scalar,
    when,
)
from narwhals.plugins import PluginName
from pizzlies._plugin.utils import is_native_frame

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from typing_extensions import TypeIs

    from narwhals.typing import FileSource, IntoDType, IntoSchema, _2DArray
    from narwhals.utils import Version
    from pizzlies._plugin.lazy import LazyPlan
    from pizzlies._plugin.namespace import DictNamespace
    from pizzlies._plugin.typing import DictFrame

__version__ = "0.1.0"

BACKEND = PluginName("pizzlies")
"""This package's plugin name, for any Narwhals function which takes a `backend`."""


def from_dict(
    data: Mapping[str, Any],
    schema: IntoSchema | Mapping[str, IntoDType | None] | None = None,
) -> DataFrame[Any]:
    """Instantiate a DataFrame from a dictionary of columns, see `narwhals.from_dict`."""
    return nw.from_dict(data, schema, backend=BACKEND)


def from_dicts(
    data: Sequence[Mapping[str, Any]],
    schema: IntoSchema | Mapping[str, IntoDType | None] | None = None,
) -> DataFrame[Any]:
    """Instantiate a DataFrame from a sequence of rows, see `narwhals.from_dicts`."""
    return nw.from_dicts(data, schema, backend=BACKEND)


def from_numpy(
    data: _2DArray, schema: IntoSchema | Sequence[str] | None = None
) -> DataFrame[Any]:
    """Instantiate a DataFrame from a 2D NumPy array, see `narwhals.from_numpy`."""
    return nw.from_numpy(data, schema, backend=BACKEND)


def from_arrow(native_frame: Any) -> DataFrame[Any]:
    """Instantiate a DataFrame from an Arrow table (anything with `__arrow_c_stream__`)."""
    return nw.from_arrow(native_frame, backend=BACKEND)


def new_series(name: str, values: Any, dtype: IntoDType | None = None) -> Series[Any]:
    """Instantiate a Series from an iterable of values, see `narwhals.new_series`."""
    return nw.new_series(name, values, dtype, backend=BACKEND)


def read_csv(
    source: FileSource, *, separator: str = ",", **kwargs: Any
) -> DataFrame[Any]:
    """Read a CSV file into a DataFrame, see `narwhals.read_csv`."""
    return nw.read_csv(source, backend=BACKEND, separator=separator, **kwargs)


def read_parquet(source: FileSource, **kwargs: Any) -> DataFrame[Any]:
    """Read a Parquet file into a DataFrame, see `narwhals.read_parquet`."""
    return nw.read_parquet(source, backend=BACKEND, **kwargs)


def scan_parquet(source: FileSource, **kwargs: Any) -> LazyFrame[Any]:
    """Read a Parquet file into a LazyFrame, see `narwhals.scan_parquet`."""
    return nw.scan_parquet(source, backend=BACKEND, **kwargs)


def scan_csv(
    source: FileSource, *, separator: str = ",", **kwargs: Any
) -> LazyFrame[Any]:
    """Read a CSV file into a LazyFrame, see `narwhals.scan_csv`."""
    return nw.scan_csv(source, backend=BACKEND, separator=separator, **kwargs)


# --- Narwhals plugin hooks -----------------------------------------------------------

NATIVE_PACKAGE = "builtins"


def __narwhals_namespace__(version: Version) -> DictNamespace:  # noqa: N807
    from pizzlies._plugin.namespace import DictNamespace

    return DictNamespace(version=version)


def is_native(native_object: object) -> TypeIs[DictFrame | LazyPlan]:
    from pizzlies._plugin.lazy import LazyPlan

    return is_native_frame(native_object) or isinstance(native_object, LazyPlan)


__all__ = [
    "BACKEND",
    "Array",
    "Binary",
    "Boolean",
    "Categorical",
    "DataFrame",
    "Date",
    "Datetime",
    "Decimal",
    "Duration",
    "Enum",
    "Expr",
    "Field",
    "Float16",
    "Float32",
    "Float64",
    "Implementation",
    "Int8",
    "Int16",
    "Int32",
    "Int64",
    "Int128",
    "LazyFrame",
    "List",
    "Map",
    "Object",
    "Schema",
    "Series",
    "String",
    "Struct",
    "Time",
    "UInt8",
    "UInt16",
    "UInt32",
    "UInt64",
    "UInt128",
    "Unknown",
    "all",
    "all_horizontal",
    "any_horizontal",
    "coalesce",
    "col",
    "concat",
    "concat_str",
    "corr",
    "cov",
    "dependencies",
    "dtypes",
    "exceptions",
    "exclude",
    "format",
    "from_arrow",
    "from_dict",
    "from_dicts",
    "from_native",
    "from_numpy",
    "generate_temporary_column_name",
    "get_native_namespace",
    "is_ordered_categorical",
    "len",
    "list",
    "lit",
    "max",
    "max_horizontal",
    "maybe_align_index",
    "maybe_convert_dtypes",
    "maybe_get_index",
    "maybe_reset_index",
    "maybe_set_index",
    "mean",
    "mean_horizontal",
    "median",
    "min",
    "min_horizontal",
    "narwhalify",
    "new_series",
    "nth",
    "read_csv",
    "read_parquet",
    "scan_csv",
    "scan_parquet",
    "selectors",
    "show_versions",
    "struct",
    "sum",
    "sum_horizontal",
    "to_native",
    "to_py_scalar",
    "when",
]
