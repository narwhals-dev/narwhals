from __future__ import annotations

import platform
import sys
from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable
from typing import Literal
from typing import Protocol
from typing import Sequence
from typing import TypeVar
from typing import Union
from typing import overload

from narwhals._expression_parsing import extract_compliant
from narwhals._pandas_like.utils import broadcast_align_and_extract_native
from narwhals.dataframe import DataFrame
from narwhals.dataframe import LazyFrame
from narwhals.dependencies import is_numpy_array
from narwhals.expr import Expr
from narwhals.translate import from_native
from narwhals.utils import Implementation
from narwhals.utils import Version
from narwhals.utils import flatten
from narwhals.utils import parse_version
from narwhals.utils import validate_laziness

# Missing type parameters for generic type "DataFrame"
# However, trying to provide one results in mypy still complaining...
# The rest of the annotations seem to work fine with this anyway
FrameT = TypeVar("FrameT", bound=Union[DataFrame, LazyFrame])  # type: ignore[type-arg]


if TYPE_CHECKING:
    from types import ModuleType

    import numpy as np

    from narwhals.dtypes import DType
    from narwhals.schema import Schema
    from narwhals.series import Series
    from narwhals.typing import IntoDataFrameT
    from narwhals.typing import IntoExpr
    from narwhals.typing import IntoFrameT
    from narwhals.typing import IntoSeriesT

    class ArrowStreamExportable(Protocol):
        def __arrow_c_stream__(
            self, requested_schema: object | None = None
        ) -> object: ...


@overload
def concat(
    items: Iterable[DataFrame[IntoDataFrameT]],
    *,
    how: Literal["horizontal", "vertical", "diagonal"] = "vertical",
) -> DataFrame[IntoDataFrameT]: ...


@overload
def concat(
    items: Iterable[LazyFrame[IntoFrameT]],
    *,
    how: Literal["horizontal", "vertical", "diagonal"] = "vertical",
) -> LazyFrame[IntoFrameT]: ...


@overload
def concat(
    items: Iterable[DataFrame[IntoDataFrameT] | LazyFrame[IntoFrameT]],
    *,
    how: Literal["horizontal", "vertical", "diagonal"] = "vertical",
) -> DataFrame[IntoDataFrameT] | LazyFrame[IntoFrameT]: ...


def concat(
    items: Iterable[DataFrame[IntoDataFrameT] | LazyFrame[IntoFrameT]],
    *,
    how: Literal["horizontal", "vertical", "diagonal"] = "vertical",
) -> DataFrame[IntoDataFrameT] | LazyFrame[IntoFrameT]:
    """Concatenate multiple DataFrames, LazyFrames into a single entity.

    Arguments:
        items: DataFrames, LazyFrames to concatenate.
        how: concatenating strategy:

            - vertical: Concatenate vertically. Column names must match.
            - horizontal: Concatenate horizontally. If lengths don't match, then
                missing rows are filled with null values.
            - diagonal: Finds a union between the column schemas and fills missing column
                values with null.

    Returns:
        A new DataFrame, Lazyframe resulting from the concatenation.

    Raises:
        TypeError: The items to concatenate should either all be eager, or all lazy
    """
    if how not in {"horizontal", "vertical", "diagonal"}:  # pragma: no cover
        msg = "Only vertical, horizontal and diagonal concatenations are supported."
        raise NotImplementedError(msg)
    if not items:
        msg = "No items to concatenate"
        raise ValueError(msg)
    items = list(items)
    validate_laziness(items)
    first_item = items[0]
    plx = first_item.__narwhals_namespace__()
    return first_item._from_compliant_dataframe(
        plx.concat([df._compliant_frame for df in items], how=how),
    )


def new_series(
    name: str,
    values: Any,
    dtype: DType | type[DType] | None = None,
    *,
    native_namespace: ModuleType,
) -> Series[Any]:
    """Instantiate Narwhals Series from iterable (e.g. list or array).

    Arguments:
        name: Name of resulting Series.
        values: Values of make Series from.
        dtype: (Narwhals) dtype. If not provided, the native library
            may auto-infer it from `values`.
        native_namespace: The native library to use for DataFrame creation.

    Returns:
        A new Series
    """
    return _new_series_impl(
        name,
        values,
        dtype,
        native_namespace=native_namespace,
        version=Version.MAIN,
    )


def _new_series_impl(
    name: str,
    values: Any,
    dtype: DType | type[DType] | None = None,
    *,
    native_namespace: ModuleType,
    version: Version,
) -> Series[Any]:
    implementation = Implementation.from_native_namespace(native_namespace)

    if implementation is Implementation.POLARS:
        if dtype:
            from narwhals._polars.utils import (
                narwhals_to_native_dtype as polars_narwhals_to_native_dtype,
            )

            dtype_pl = polars_narwhals_to_native_dtype(dtype, version=version)
        else:
            dtype_pl = None

        native_series = native_namespace.Series(name=name, values=values, dtype=dtype_pl)
    elif implementation in {
        Implementation.PANDAS,
        Implementation.MODIN,
        Implementation.CUDF,
    }:
        if dtype:
            from narwhals._pandas_like.utils import (
                narwhals_to_native_dtype as pandas_like_narwhals_to_native_dtype,
            )

            backend_version = parse_version(native_namespace.__version__)
            dtype = pandas_like_narwhals_to_native_dtype(
                dtype, None, implementation, backend_version, version
            )
        native_series = native_namespace.Series(values, name=name, dtype=dtype)

    elif implementation is Implementation.PYARROW:
        if dtype:
            from narwhals._arrow.utils import (
                narwhals_to_native_dtype as arrow_narwhals_to_native_dtype,
            )

            dtype = arrow_narwhals_to_native_dtype(dtype, version=version)
        native_series = native_namespace.chunked_array([values], type=dtype)

    elif implementation is Implementation.DASK:
        msg = "Dask support in Narwhals is lazy-only, so `new_series` is " "not supported"
        raise NotImplementedError(msg)
    else:  # pragma: no cover
        try:
            # implementation is UNKNOWN, Narwhals extension using this feature should
            # implement `from_dict` function in the top-level namespace.
            native_series = native_namespace.new_series(name, values, dtype)
        except AttributeError as e:
            msg = "Unknown namespace is expected to implement `Series` constructor."
            raise AttributeError(msg) from e
    return from_native(native_series, series_only=True).alias(name)


def from_dict(
    data: dict[str, Any],
    schema: dict[str, DType] | Schema | None = None,
    *,
    native_namespace: ModuleType | None = None,
) -> DataFrame[Any]:
    """Instantiate DataFrame from dictionary.

    Indexes (if present, for pandas-like backends) are aligned following
    the [left-hand-rule](../pandas_like_concepts/pandas_index.md/).

    Notes:
        For pandas-like dataframes, conversion to schema is applied after dataframe
        creation.

    Arguments:
        data: Dictionary to create DataFrame from.
        schema: The DataFrame schema as Schema or dict of {name: type}.
        native_namespace: The native library to use for DataFrame creation. Only
            necessary if inputs are not Narwhals Series.

    Returns:
        A new DataFrame.
    """
    return _from_dict_impl(
        data,
        schema,
        native_namespace=native_namespace,
        version=Version.MAIN,
    )


def _from_dict_impl(
    data: dict[str, Any],
    schema: dict[str, DType] | Schema | None = None,
    *,
    native_namespace: ModuleType | None = None,
    version: Version,
) -> DataFrame[Any]:
    from narwhals.series import Series
    from narwhals.translate import to_native

    if not data:
        msg = "from_dict cannot be called with empty dictionary"
        raise ValueError(msg)
    if native_namespace is None:
        for val in data.values():
            if isinstance(val, Series):
                native_namespace = val.__native_namespace__()
                break
        else:
            msg = "Calling `from_dict` without `native_namespace` is only supported if all input values are already Narwhals Series"
            raise TypeError(msg)
        data = {key: to_native(value, pass_through=True) for key, value in data.items()}
    implementation = Implementation.from_native_namespace(native_namespace)

    if implementation is Implementation.POLARS:
        if schema:
            from narwhals._polars.utils import (
                narwhals_to_native_dtype as polars_narwhals_to_native_dtype,
            )

            schema_pl = {
                name: polars_narwhals_to_native_dtype(dtype, version=version)
                for name, dtype in schema.items()
            }
        else:
            schema_pl = None

        native_frame = native_namespace.from_dict(data, schema=schema_pl)
    elif implementation in {
        Implementation.PANDAS,
        Implementation.MODIN,
        Implementation.CUDF,
    }:
        aligned_data = {}
        left_most_series = None
        for key, native_series in data.items():
            if isinstance(native_series, native_namespace.Series):
                compliant_series = from_native(
                    native_series, series_only=True
                )._compliant_series
                if left_most_series is None:
                    left_most_series = compliant_series
                    aligned_data[key] = native_series
                else:
                    aligned_data[key] = broadcast_align_and_extract_native(
                        left_most_series, compliant_series
                    )[1]
            else:
                aligned_data[key] = native_series

        native_frame = native_namespace.DataFrame.from_dict(aligned_data)

        if schema:
            from narwhals._pandas_like.utils import (
                narwhals_to_native_dtype as pandas_like_narwhals_to_native_dtype,
            )

            backend_version = parse_version(native_namespace.__version__)
            schema = {
                name: pandas_like_narwhals_to_native_dtype(
                    schema[name], native_type, implementation, backend_version, version
                )
                for name, native_type in native_frame.dtypes.items()
            }
            native_frame = native_frame.astype(schema)

    elif implementation is Implementation.PYARROW:
        if schema:
            from narwhals._arrow.utils import (
                narwhals_to_native_dtype as arrow_narwhals_to_native_dtype,
            )

            schema = native_namespace.schema(
                [
                    (name, arrow_narwhals_to_native_dtype(dtype, version))
                    for name, dtype in schema.items()
                ]
            )
        native_frame = native_namespace.table(data, schema=schema)
    else:  # pragma: no cover
        try:
            # implementation is UNKNOWN, Narwhals extension using this feature should
            # implement `from_dict` function in the top-level namespace.
            native_frame = native_namespace.from_dict(data, schema=schema)
        except AttributeError as e:
            msg = "Unknown namespace is expected to implement `from_dict` function."
            raise AttributeError(msg) from e
    return from_native(native_frame, eager_only=True)


def from_numpy(
    data: np.ndarray,
    schema: dict[str, DType] | Schema | list[str] | None = None,
    *,
    native_namespace: ModuleType,
) -> DataFrame[Any]:
    """Construct a DataFrame from a NumPy ndarray.

    Notes:
        Only row orientation is currently supported.

        For pandas-like dataframes, conversion to schema is applied after dataframe
        creation.

    Arguments:
        data: Two-dimensional data represented as a NumPy ndarray.
        schema: The DataFrame schema as Schema, dict of {name: type}, or a list of str.
        native_namespace: The native library to use for DataFrame creation.

    Returns:
        A new DataFrame.
    """
    return _from_numpy_impl(
        data,
        schema,
        native_namespace=native_namespace,
        version=Version.MAIN,
    )


def _from_numpy_impl(
    data: np.ndarray,
    schema: dict[str, DType] | Schema | list[str] | None = None,
    *,
    native_namespace: ModuleType,
    version: Version,
) -> DataFrame[Any]:
    from narwhals.schema import Schema

    if data.ndim != 2:
        msg = "`from_numpy` only accepts 2D numpy arrays"
        raise ValueError(msg)
    implementation = Implementation.from_native_namespace(native_namespace)

    if implementation is Implementation.POLARS:
        if isinstance(schema, (dict, Schema)):
            from narwhals._polars.utils import (
                narwhals_to_native_dtype as polars_narwhals_to_native_dtype,
            )

            schema = {
                name: polars_narwhals_to_native_dtype(dtype, version=version)  # type: ignore[misc]
                for name, dtype in schema.items()
            }
        elif schema is None:
            native_frame = native_namespace.from_numpy(data)
        elif not isinstance(schema, list):
            msg = (
                "`schema` is expected to be one of the following types: "
                "dict[str, DType] | Schema | list[str]. "
                f"Got {type(schema)}."
            )
            raise TypeError(msg)
        native_frame = native_namespace.from_numpy(data, schema=schema)

    elif implementation in {
        Implementation.PANDAS,
        Implementation.MODIN,
        Implementation.CUDF,
    }:
        if isinstance(schema, (dict, Schema)):
            from narwhals._pandas_like.utils import (
                narwhals_to_native_dtype as pandas_like_narwhals_to_native_dtype,
            )

            backend_version = parse_version(native_namespace.__version__)
            schema = {
                name: pandas_like_narwhals_to_native_dtype(
                    schema[name], native_type, implementation, backend_version, version
                )
                for name, native_type in schema.items()
            }
            native_frame = native_namespace.DataFrame(data, columns=schema.keys()).astype(
                schema
            )
        elif isinstance(schema, list):
            native_frame = native_namespace.DataFrame(data, columns=schema)
        elif schema is None:
            native_frame = native_namespace.DataFrame(
                data, columns=["column_" + str(x) for x in range(data.shape[1])]
            )
        else:
            msg = (
                "`schema` is expected to be one of the following types: "
                "dict[str, DType] | Schema | list[str]. "
                f"Got {type(schema)}."
            )
            raise TypeError(msg)

    elif implementation is Implementation.PYARROW:
        pa_arrays = [native_namespace.array(val) for val in data.T]
        if isinstance(schema, (dict, Schema)):
            from narwhals._arrow.utils import (
                narwhals_to_native_dtype as arrow_narwhals_to_native_dtype,
            )

            schema = native_namespace.schema(
                [
                    (name, arrow_narwhals_to_native_dtype(dtype, version))
                    for name, dtype in schema.items()
                ]
            )
            native_frame = native_namespace.Table.from_arrays(pa_arrays, schema=schema)
        elif isinstance(schema, list):
            native_frame = native_namespace.Table.from_arrays(pa_arrays, names=schema)
        elif schema is None:
            native_frame = native_namespace.Table.from_arrays(
                pa_arrays, names=["column_" + str(x) for x in range(data.shape[1])]
            )
        else:
            msg = (
                "`schema` is expected to be one of the following types: "
                "dict[str, DType] | Schema | list[str]. "
                f"Got {type(schema)}."
            )
            raise TypeError(msg)
    else:  # pragma: no cover
        try:
            # implementation is UNKNOWN, Narwhals extension using this feature should
            # implement `from_numpy` function in the top-level namespace.
            native_frame = native_namespace.from_numpy(data, schema=schema)
        except AttributeError as e:
            msg = "Unknown namespace is expected to implement `from_numpy` function."
            raise AttributeError(msg) from e
    return from_native(native_frame, eager_only=True)


def from_arrow(
    native_frame: ArrowStreamExportable, *, native_namespace: ModuleType
) -> DataFrame[Any]:
    """Construct a DataFrame from an object which supports the PyCapsule Interface.

    Arguments:
        native_frame: Object which implements `__arrow_c_stream__`.
        native_namespace: The native library to use for DataFrame creation.

    Returns:
        A new DataFrame.
    """
    if not hasattr(native_frame, "__arrow_c_stream__"):
        msg = f"Given object of type {type(native_frame)} does not support PyCapsule interface"
        raise TypeError(msg)
    implementation = Implementation.from_native_namespace(native_namespace)

    if implementation is Implementation.POLARS and parse_version(
        native_namespace.__version__
    ) >= (1, 3):
        native_frame = native_namespace.DataFrame(native_frame)
    elif implementation in {
        Implementation.PANDAS,
        Implementation.MODIN,
        Implementation.CUDF,
        Implementation.POLARS,
    }:
        # These don't (yet?) support the PyCapsule Interface for import
        # so we go via PyArrow
        try:
            import pyarrow as pa  # ignore-banned-import
        except ModuleNotFoundError as exc:  # pragma: no cover
            msg = f"PyArrow>=14.0.0 is required for `from_arrow` for object of type {native_namespace}"
            raise ModuleNotFoundError(msg) from exc
        if parse_version(pa.__version__) < (14, 0):  # pragma: no cover
            msg = f"PyArrow>=14.0.0 is required for `from_arrow` for object of type {native_namespace}"
            raise ModuleNotFoundError(msg) from None

        tbl = pa.table(native_frame)
        if implementation is Implementation.PANDAS:
            native_frame = tbl.to_pandas()
        elif implementation is Implementation.MODIN:  # pragma: no cover
            from modin.pandas.utils import from_arrow

            native_frame = from_arrow(tbl)
        elif implementation is Implementation.CUDF:  # pragma: no cover
            native_frame = native_namespace.DataFrame.from_arrow(tbl)
        elif implementation is Implementation.POLARS:  # pragma: no cover
            native_frame = native_namespace.from_arrow(tbl)
        else:  # pragma: no cover
            msg = "congratulations, you entered unrecheable code - please report a bug"
            raise AssertionError(msg)
    elif implementation is Implementation.PYARROW:
        native_frame = native_namespace.table(native_frame)
    else:  # pragma: no cover
        try:
            # implementation is UNKNOWN, Narwhals extension using this feature should
            # implement PyCapsule support
            native_frame = native_namespace.DataFrame(native_frame)
        except AttributeError as e:
            msg = "Unknown namespace is expected to implement `DataFrame` class which accepts object which supports PyCapsule Interface."
            raise AttributeError(msg) from e
    return from_native(native_frame, eager_only=True)


def _get_sys_info() -> dict[str, str]:
    """System information.

    Returns system and Python version information

    Copied from sklearn

    Returns:
        Dictionary with system info.
    """
    python = sys.version.replace("\n", " ")

    blob = (
        ("python", python),
        ("executable", sys.executable),
        ("machine", platform.platform()),
    )

    return dict(blob)


def _get_deps_info() -> dict[str, str]:
    """Overview of the installed version of main dependencies.

    This function does not import the modules to collect the version numbers
    but instead relies on standard Python package metadata.

    Returns version information on relevant Python libraries

    This function and show_versions were copied from sklearn and adapted

    Returns:
        Mapping from dependency to version.
    """
    deps = (
        "pandas",
        "polars",
        "cudf",
        "modin",
        "pyarrow",
        "numpy",
    )

    from . import __version__

    deps_info = {
        "narwhals": __version__,
    }

    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version

    for modname in deps:
        try:
            deps_info[modname] = version(modname)
        except PackageNotFoundError:  # noqa: PERF203
            deps_info[modname] = ""
    return deps_info


def show_versions() -> None:
    """Print useful debugging information."""
    sys_info = _get_sys_info()
    deps_info = _get_deps_info()

    print("\nSystem:")  # noqa: T201
    for k, stat in sys_info.items():
        print(f"{k:>10}: {stat}")  # noqa: T201

    print("\nPython dependencies:")  # noqa: T201
    for k, stat in deps_info.items():
        print(f"{k:>13}: {stat}")  # noqa: T201


def get_level(
    obj: DataFrame[Any] | LazyFrame[Any] | Series[IntoSeriesT],
) -> Literal["full", "lazy", "interchange"]:
    """Level of support Narwhals has for current object.

    Arguments:
        obj: Dataframe or Series.

    Returns:
        This can be one of:

            - 'full': full Narwhals API support
            - 'lazy': only lazy operations are supported. This excludes anything
              which involves iterating over rows in Python.
            - 'interchange': only metadata operations are supported (`df.schema`)
    """
    return obj._level


def read_csv(
    source: str,
    *,
    native_namespace: ModuleType,
    **kwargs: Any,
) -> DataFrame[Any]:
    """Read a CSV file into a DataFrame.

    Arguments:
        source: Path to a file.
        native_namespace: The native library to use for DataFrame creation.
        kwargs: Extra keyword arguments which are passed to the native CSV reader.
            For example, you could use
            `nw.read_csv('file.csv', native_namespace=pd, engine='pyarrow')`.

    Returns:
        DataFrame.
    """
    return _read_csv_impl(source, native_namespace=native_namespace, **kwargs)


def _read_csv_impl(
    source: str, *, native_namespace: ModuleType, **kwargs: Any
) -> DataFrame[Any]:
    implementation = Implementation.from_native_namespace(native_namespace)
    if implementation in (
        Implementation.POLARS,
        Implementation.PANDAS,
        Implementation.MODIN,
        Implementation.CUDF,
    ):
        native_frame = native_namespace.read_csv(source, **kwargs)
    elif implementation is Implementation.PYARROW:
        from pyarrow import csv  # ignore-banned-import

        native_frame = csv.read_csv(source, **kwargs)
    else:  # pragma: no cover
        try:
            # implementation is UNKNOWN, Narwhals extension using this feature should
            # implement `read_csv` function in the top-level namespace.
            native_frame = native_namespace.read_csv(source=source, **kwargs)
        except AttributeError as e:
            msg = "Unknown namespace is expected to implement `read_csv` function."
            raise AttributeError(msg) from e
    return from_native(native_frame, eager_only=True)


def scan_csv(
    source: str, *, native_namespace: ModuleType, **kwargs: Any
) -> LazyFrame[Any]:
    """Lazily read from a CSV file.

    For the libraries that do not support lazy dataframes, the function reads
    a csv file eagerly and then converts the resulting dataframe to a lazyframe.

    Arguments:
        source: Path to a file.
        native_namespace: The native library to use for DataFrame creation.
        kwargs: Extra keyword arguments which are passed to the native CSV reader.
            For example, you could use
            `nw.scan_csv('file.csv', native_namespace=pd, engine='pyarrow')`.

    Returns:
        LazyFrame.
    """
    return _scan_csv_impl(source, native_namespace=native_namespace, **kwargs)


def _scan_csv_impl(
    source: str, *, native_namespace: ModuleType, **kwargs: Any
) -> LazyFrame[Any]:
    implementation = Implementation.from_native_namespace(native_namespace)
    if implementation is Implementation.POLARS:
        native_frame = native_namespace.scan_csv(source, **kwargs)
    elif implementation in (
        Implementation.PANDAS,
        Implementation.MODIN,
        Implementation.CUDF,
        Implementation.DASK,
        Implementation.DUCKDB,
    ):
        native_frame = native_namespace.read_csv(source, **kwargs)
    elif implementation is Implementation.PYARROW:
        from pyarrow import csv  # ignore-banned-import

        native_frame = csv.read_csv(source, **kwargs)
    else:  # pragma: no cover
        try:
            # implementation is UNKNOWN, Narwhals extension using this feature should
            # implement `scan_csv` function in the top-level namespace.
            native_frame = native_namespace.scan_csv(source=source, **kwargs)
        except AttributeError as e:
            msg = "Unknown namespace is expected to implement `scan_csv` function."
            raise AttributeError(msg) from e
    return from_native(native_frame).lazy()


def read_parquet(
    source: str,
    *,
    native_namespace: ModuleType,
    **kwargs: Any,
) -> DataFrame[Any]:
    """Read into a DataFrame from a parquet file.

    Arguments:
        source: Path to a file.
        native_namespace: The native library to use for DataFrame creation.
        kwargs: Extra keyword arguments which are passed to the native parquet reader.
            For example, you could use
            `nw.read_parquet('file.parquet', native_namespace=pd, engine='pyarrow')`.

    Returns:
        DataFrame.
    """
    return _read_parquet_impl(source, native_namespace=native_namespace, **kwargs)


def _read_parquet_impl(
    source: str, *, native_namespace: ModuleType, **kwargs: Any
) -> DataFrame[Any]:
    implementation = Implementation.from_native_namespace(native_namespace)
    if implementation in (
        Implementation.POLARS,
        Implementation.PANDAS,
        Implementation.MODIN,
        Implementation.CUDF,
        Implementation.DUCKDB,
    ):
        native_frame = native_namespace.read_parquet(source, **kwargs)
    elif implementation is Implementation.PYARROW:
        import pyarrow.parquet as pq  # ignore-banned-import

        native_frame = pq.read_table(source, **kwargs)
    else:  # pragma: no cover
        try:
            # implementation is UNKNOWN, Narwhals extension using this feature should
            # implement `read_parquet` function in the top-level namespace.
            native_frame = native_namespace.read_parquet(source=source, **kwargs)
        except AttributeError as e:
            msg = "Unknown namespace is expected to implement `read_parquet` function."
            raise AttributeError(msg) from e
    return from_native(native_frame, eager_only=True)


def scan_parquet(
    source: str, *, native_namespace: ModuleType, **kwargs: Any
) -> LazyFrame[Any]:
    """Lazily read from a parquet file.

    For the libraries that do not support lazy dataframes, the function reads
    a parquet file eagerly and then converts the resulting dataframe to a lazyframe.

    Arguments:
        source: Path to a file.
        native_namespace: The native library to use for DataFrame creation.
        kwargs: Extra keyword arguments which are passed to the native parquet reader.
            For example, you could use
            `nw.scan_parquet('file.parquet', native_namespace=pd, engine='pyarrow')`.

    Returns:
        LazyFrame.
    """
    return _scan_parquet_impl(source, native_namespace=native_namespace, **kwargs)


def _scan_parquet_impl(
    source: str, *, native_namespace: ModuleType, **kwargs: Any
) -> LazyFrame[Any]:
    implementation = Implementation.from_native_namespace(native_namespace)
    if implementation is Implementation.POLARS:
        native_frame = native_namespace.scan_parquet(source, **kwargs)
    elif implementation in (
        Implementation.PANDAS,
        Implementation.MODIN,
        Implementation.CUDF,
        Implementation.DASK,
        Implementation.DUCKDB,
    ):
        native_frame = native_namespace.read_parquet(source, **kwargs)
    elif implementation is Implementation.PYARROW:
        import pyarrow.parquet as pq  # ignore-banned-import

        native_frame = pq.read_table(source, **kwargs)
    else:  # pragma: no cover
        try:
            # implementation is UNKNOWN, Narwhals extension using this feature should
            # implement `scan_parquet` function in the top-level namespace.
            native_frame = native_namespace.scan_parquet(source=source, **kwargs)
        except AttributeError as e:
            msg = "Unknown namespace is expected to implement `scan_parquet` function."
            raise AttributeError(msg) from e
    return from_native(native_frame).lazy()


def col(*names: str | Iterable[str]) -> Expr:
    """Creates an expression that references one or more columns by their name(s).

    Arguments:
        names: Name(s) of the columns to use.

    Returns:
        A new expression.
    """

    def func(plx: Any) -> Any:
        return plx.col(*flatten(names))

    return Expr(func)


def nth(*indices: int | Sequence[int]) -> Expr:
    """Creates an expression that references one or more columns by their index(es).

    Notes:
        `nth` is not supported for Polars version<1.0.0. Please use
        [`narwhals.col`][] instead.

    Arguments:
        indices: One or more indices representing the columns to retrieve.

    Returns:
        A new expression.
    """

    def func(plx: Any) -> Any:
        return plx.nth(*flatten(indices))

    return Expr(func)


# Add underscore so it doesn't conflict with builtin `all`
def all_() -> Expr:
    """Instantiate an expression representing all columns.

    Returns:
        A new expression.
    """
    return Expr(lambda plx: plx.all())


# Add underscore so it doesn't conflict with builtin `len`
def len_() -> Expr:
    """Return the number of rows.

    Returns:
        A new expression.
    """

    def func(plx: Any) -> Any:
        return plx.len()

    return Expr(func)


def sum(*columns: str) -> Expr:
    """Sum all values.

    Note:
        Syntactic sugar for ``nw.col(columns).sum()``

    Arguments:
        columns: Name(s) of the columns to use in the aggregation function

    Returns:
        A new expression.
    """
    return Expr(lambda plx: plx.col(*columns).sum())


def mean(*columns: str) -> Expr:
    """Get the mean value.

    Note:
        Syntactic sugar for ``nw.col(columns).mean()``

    Arguments:
        columns: Name(s) of the columns to use in the aggregation function

    Returns:
        A new expression.
    """
    return Expr(lambda plx: plx.col(*columns).mean())


def median(*columns: str) -> Expr:
    """Get the median value.

    Notes:
        - Syntactic sugar for ``nw.col(columns).median()``
        - Results might slightly differ across backends due to differences in the
            underlying algorithms used to compute the median.

    Arguments:
        columns: Name(s) of the columns to use in the aggregation function

    Returns:
        A new expression.
    """
    return Expr(lambda plx: plx.col(*columns).median())


def min(*columns: str) -> Expr:
    """Return the minimum value.

    Note:
       Syntactic sugar for ``nw.col(columns).min()``.

    Arguments:
        columns: Name(s) of the columns to use in the aggregation function.

    Returns:
        A new expression.
    """
    return Expr(lambda plx: plx.col(*columns).min())


def max(*columns: str) -> Expr:
    """Return the maximum value.

    Note:
       Syntactic sugar for ``nw.col(columns).max()``.

    Arguments:
        columns: Name(s) of the columns to use in the aggregation function.

    Returns:
        A new expression.
    """
    return Expr(lambda plx: plx.col(*columns).max())


def sum_horizontal(*exprs: IntoExpr | Iterable[IntoExpr]) -> Expr:
    """Sum all values horizontally across columns.

    Warning:
        Unlike Polars, we support horizontal sum over numeric columns only.

    Arguments:
        exprs: Name(s) of the columns to use in the aggregation function. Accepts
            expression input.

    Returns:
        A new expression.
    """
    if not exprs:
        msg = "At least one expression must be passed to `sum_horizontal`"
        raise ValueError(msg)
    return Expr(
        lambda plx: plx.sum_horizontal(
            *[extract_compliant(plx, v) for v in flatten(exprs)]
        )
    )


def min_horizontal(*exprs: IntoExpr | Iterable[IntoExpr]) -> Expr:
    """Get the minimum value horizontally across columns.

    Notes:
        We support `min_horizontal` over numeric columns only.

    Arguments:
        exprs: Name(s) of the columns to use in the aggregation function. Accepts
            expression input.

    Returns:
        A new expression.
    """
    if not exprs:
        msg = "At least one expression must be passed to `min_horizontal`"
        raise ValueError(msg)
    return Expr(
        lambda plx: plx.min_horizontal(
            *[extract_compliant(plx, v) for v in flatten(exprs)]
        )
    )


def max_horizontal(*exprs: IntoExpr | Iterable[IntoExpr]) -> Expr:
    """Get the maximum value horizontally across columns.

    Notes:
        We support `max_horizontal` over numeric columns only.

    Arguments:
        exprs: Name(s) of the columns to use in the aggregation function. Accepts
            expression input.

    Returns:
        A new expression.
    """
    if not exprs:
        msg = "At least one expression must be passed to `max_horizontal`"
        raise ValueError(msg)
    return Expr(
        lambda plx: plx.max_horizontal(
            *[extract_compliant(plx, v) for v in flatten(exprs)]
        )
    )


class When:
    def __init__(self, *predicates: IntoExpr | Iterable[IntoExpr]) -> None:
        self._predicates = flatten([predicates])
        if not self._predicates:
            msg = "At least one predicate needs to be provided to `narwhals.when`."
            raise TypeError(msg)

    def _extract_predicates(self, plx: Any) -> Any:
        return [extract_compliant(plx, v) for v in self._predicates]

    def then(self, value: Any) -> Then:
        return Then(
            lambda plx: plx.when(*self._extract_predicates(plx)).then(
                extract_compliant(plx, value)
            )
        )


class Then(Expr):
    def otherwise(self, value: Any) -> Expr:
        return Expr(
            lambda plx: self._to_compliant_expr(plx).otherwise(
                extract_compliant(plx, value)
            )
        )


def when(*predicates: IntoExpr | Iterable[IntoExpr]) -> When:
    """Start a `when-then-otherwise` expression.

    Expression similar to an `if-else` statement in Python. Always initiated by a
    `pl.when(<condition>).then(<value if condition>)`, and optionally followed by
    chaining one or more `.when(<condition>).then(<value>)` statements.
    Chained when-then operations should be read as Python `if, elif, ... elif`
    blocks, not as `if, if, ... if`, i.e. the first condition that evaluates to
    `True` will be picked.
    If none of the conditions are `True`, an optional
    `.otherwise(<value if all statements are false>)` can be appended at the end.
    If not appended, and none of the conditions are `True`, `None` will be returned.

    Arguments:
        predicates: Condition(s) that must be met in order to apply the subsequent
            statement. Accepts one or more boolean expressions, which are implicitly
            combined with `&`. String input is parsed as a column name.

    Returns:
        A "when" object, which `.then` can be called on.
    """
    return When(*predicates)


def all_horizontal(*exprs: IntoExpr | Iterable[IntoExpr]) -> Expr:
    r"""Compute the bitwise AND horizontally across columns.

    Arguments:
        exprs: Name(s) of the columns to use in the aggregation function. Accepts
            expression input.

    Returns:
        A new expression.
    """
    if not exprs:
        msg = "At least one expression must be passed to `all_horizontal`"
        raise ValueError(msg)
    return Expr(
        lambda plx: plx.all_horizontal(
            *[extract_compliant(plx, v) for v in flatten(exprs)]
        )
    )


def lit(value: Any, dtype: DType | type[DType] | None = None) -> Expr:
    """Return an expression representing a literal value.

    Arguments:
        value: The value to use as literal.
        dtype: The data type of the literal value. If not provided, the data type will
            be inferred.

    Returns:
        A new expression.
    """
    if is_numpy_array(value):
        msg = (
            "numpy arrays are not supported as literal values. "
            "Consider using `with_columns` to create a new column from the array."
        )
        raise ValueError(msg)

    if isinstance(value, (list, tuple)):
        msg = f"Nested datatypes are not supported yet. Got {value}"
        raise NotImplementedError(msg)

    return Expr(lambda plx: plx.lit(value, dtype))


def any_horizontal(*exprs: IntoExpr | Iterable[IntoExpr]) -> Expr:
    r"""Compute the bitwise OR horizontally across columns.

    Arguments:
        exprs: Name(s) of the columns to use in the aggregation function. Accepts
            expression input.

    Returns:
        A new expression.
    """
    if not exprs:
        msg = "At least one expression must be passed to `any_horizontal`"
        raise ValueError(msg)
    return Expr(
        lambda plx: plx.any_horizontal(
            *[extract_compliant(plx, v) for v in flatten(exprs)]
        )
    )


def mean_horizontal(*exprs: IntoExpr | Iterable[IntoExpr]) -> Expr:
    """Compute the mean of all values horizontally across columns.

    Arguments:
        exprs: Name(s) of the columns to use in the aggregation function. Accepts
            expression input.

    Returns:
        A new expression.
    """
    if not exprs:
        msg = "At least one expression must be passed to `mean_horizontal`"
        raise ValueError(msg)
    return Expr(
        lambda plx: plx.mean_horizontal(
            *[extract_compliant(plx, v) for v in flatten(exprs)]
        )
    )


def concat_str(
    exprs: IntoExpr | Iterable[IntoExpr],
    *more_exprs: IntoExpr,
    separator: str = "",
    ignore_nulls: bool = False,
) -> Expr:
    r"""Horizontally concatenate columns into a single string column.

    Arguments:
        exprs: Columns to concatenate into a single string column. Accepts expression
            input. Strings are parsed as column names, other non-expression inputs are
            parsed as literals. Non-`String` columns are cast to `String`.
        *more_exprs: Additional columns to concatenate into a single string column,
            specified as positional arguments.
        separator: String that will be used to separate the values of each column.
        ignore_nulls: Ignore null values (default is `False`).
            If set to `False`, null values will be propagated and if the row contains any
            null values, the output is null.

    Returns:
        A new expression.
    """
    return Expr(
        lambda plx: plx.concat_str(
            [extract_compliant(plx, v) for v in flatten([exprs])],
            *[extract_compliant(plx, v) for v in more_exprs],
            separator=separator,
            ignore_nulls=ignore_nulls,
        )
    )
