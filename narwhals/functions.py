from __future__ import annotations

import platform
import sys
from importlib.metadata import version
from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable
from typing import Literal
from typing import Mapping
from typing import Sequence
from typing import TypeVar
from typing import cast

from narwhals._expression_parsing import ExprKind
from narwhals._expression_parsing import ExprMetadata
from narwhals._expression_parsing import apply_n_ary_operation
from narwhals._expression_parsing import check_expressions_preserve_length
from narwhals._expression_parsing import combine_metadata
from narwhals._expression_parsing import extract_compliant
from narwhals._expression_parsing import is_scalar_like
from narwhals.dependencies import is_narwhals_series
from narwhals.dependencies import is_numpy_array
from narwhals.dependencies import is_numpy_array_2d
from narwhals.dependencies import is_pyarrow_table
from narwhals.exceptions import InvalidOperationError
from narwhals.expr import Expr
from narwhals.series import Series
from narwhals.translate import from_native
from narwhals.translate import to_native
from narwhals.utils import Implementation
from narwhals.utils import Version
from narwhals.utils import deprecate_native_namespace
from narwhals.utils import flatten
from narwhals.utils import is_compliant_expr
from narwhals.utils import is_eager_allowed
from narwhals.utils import is_sequence_but_not_str
from narwhals.utils import parse_version
from narwhals.utils import supports_arrow_c_stream
from narwhals.utils import validate_laziness

if TYPE_CHECKING:
    from types import ModuleType

    from typing_extensions import TypeAlias
    from typing_extensions import TypeIs

    from narwhals._compliant import CompliantExpr
    from narwhals._compliant import CompliantNamespace
    from narwhals._translate import IntoArrowTable
    from narwhals.dataframe import DataFrame
    from narwhals.dataframe import LazyFrame
    from narwhals.dtypes import DType
    from narwhals.schema import Schema
    from narwhals.series import Series
    from narwhals.typing import ConcatMethod
    from narwhals.typing import IntoExpr
    from narwhals.typing import IntoSeriesT
    from narwhals.typing import NativeFrame
    from narwhals.typing import NativeLazyFrame
    from narwhals.typing import NativeSeries
    from narwhals.typing import NonNestedLiteral
    from narwhals.typing import _1DArray
    from narwhals.typing import _2DArray

    _IntoSchema: TypeAlias = "Mapping[str, DType] | Schema | Sequence[str] | None"
    FrameT = TypeVar("FrameT", "DataFrame[Any]", "LazyFrame[Any]")


def concat(items: Iterable[FrameT], *, how: ConcatMethod = "vertical") -> FrameT:
    """Concatenate multiple DataFrames, LazyFrames into a single entity.

    Arguments:
        items: DataFrames, LazyFrames to concatenate.
        how: concatenating strategy

            - vertical: Concatenate vertically. Column names must match.
            - horizontal: Concatenate horizontally. If lengths don't match, then
                missing rows are filled with null values. This is only supported
                when all inputs are (eager) DataFrames.
            - diagonal: Finds a union between the column schemas and fills missing column
                values with null.

    Returns:
        A new DataFrame or LazyFrame resulting from the concatenation.

    Raises:
        TypeError: The items to concatenate should either all be eager, or all lazy

    Examples:
        Let's take an example of vertical concatenation:

        >>> import pandas as pd
        >>> import polars as pl
        >>> import pyarrow as pa
        >>> import narwhals as nw

        Let's look at one case a for vertical concatenation (pandas backed):

        >>> df_pd_1 = nw.from_native(pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
        >>> df_pd_2 = nw.from_native(pd.DataFrame({"a": [5, 2], "b": [1, 4]}))
        >>> nw.concat([df_pd_1, df_pd_2], how="vertical")
        ┌──────────────────┐
        |Narwhals DataFrame|
        |------------------|
        |        a  b      |
        |     0  1  4      |
        |     1  2  5      |
        |     2  3  6      |
        |     0  5  1      |
        |     1  2  4      |
        └──────────────────┘

        Let's look at one case a for horizontal concatenation (polars backed):

        >>> df_pl_1 = nw.from_native(pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
        >>> df_pl_2 = nw.from_native(pl.DataFrame({"c": [5, 2], "d": [1, 4]}))
        >>> nw.concat([df_pl_1, df_pl_2], how="horizontal")
        ┌───────────────────────────┐
        |    Narwhals DataFrame     |
        |---------------------------|
        |shape: (3, 4)              |
        |┌─────┬─────┬──────┬──────┐|
        |│ a   ┆ b   ┆ c    ┆ d    │|
        |│ --- ┆ --- ┆ ---  ┆ ---  │|
        |│ i64 ┆ i64 ┆ i64  ┆ i64  │|
        |╞═════╪═════╪══════╪══════╡|
        |│ 1   ┆ 4   ┆ 5    ┆ 1    │|
        |│ 2   ┆ 5   ┆ 2    ┆ 4    │|
        |│ 3   ┆ 6   ┆ null ┆ null │|
        |└─────┴─────┴──────┴──────┘|
        └───────────────────────────┘

        Let's look at one case a for diagonal concatenation (pyarrow backed):

        >>> df_pa_1 = nw.from_native(pa.table({"a": [1, 2], "b": [3.5, 4.5]}))
        >>> df_pa_2 = nw.from_native(pa.table({"a": [3, 4], "z": ["x", "y"]}))
        >>> nw.concat([df_pa_1, df_pa_2], how="diagonal")
        ┌──────────────────────────┐
        |    Narwhals DataFrame    |
        |--------------------------|
        |pyarrow.Table             |
        |a: int64                  |
        |b: double                 |
        |z: string                 |
        |----                      |
        |a: [[1,2],[3,4]]          |
        |b: [[3.5,4.5],[null,null]]|
        |z: [[null,null],["x","y"]]|
        └──────────────────────────┘
    """
    from narwhals.dependencies import is_narwhals_lazyframe

    if not items:
        msg = "No items to concatenate."
        raise ValueError(msg)
    items = list(items)
    validate_laziness(items)
    if how not in {"horizontal", "vertical", "diagonal"}:  # pragma: no cover
        msg = "Only vertical, horizontal and diagonal concatenations are supported."
        raise NotImplementedError(msg)
    first_item = items[0]
    if is_narwhals_lazyframe(first_item) and how == "horizontal":
        msg = (
            "Horizontal concatenation is not supported for LazyFrames.\n\n"
            "Hint: you may want to use `join` instead."
        )
        raise InvalidOperationError(msg)
    plx = first_item.__narwhals_namespace__()
    return first_item._with_compliant(
        plx.concat([df._compliant_frame for df in items], how=how),
    )


@deprecate_native_namespace(warn_version="1.31.0", required=True)
def new_series(
    name: str,
    values: Any,
    dtype: DType | type[DType] | None = None,
    *,
    backend: ModuleType | Implementation | str | None = None,
    native_namespace: ModuleType | None = None,  # noqa: ARG001
) -> Series[Any]:
    """Instantiate Narwhals Series from iterable (e.g. list or array).

    Arguments:
        name: Name of resulting Series.
        values: Values of make Series from.
        dtype: (Narwhals) dtype. If not provided, the native library
            may auto-infer it from `values`.
        backend: specifies which eager backend instantiate to.

            `backend` can be specified in various ways

            - As `Implementation.<BACKEND>` with `BACKEND` being `PANDAS`, `PYARROW`,
                `POLARS`, `MODIN` or `CUDF`.
            - As a string: `"pandas"`, `"pyarrow"`, `"polars"`, `"modin"` or `"cudf"`.
            - Directly as a module `pandas`, `pyarrow`, `polars`, `modin` or `cudf`.
        native_namespace: The native library to use for DataFrame creation.

            *Deprecated* (v1.31.0)

            Please use `backend` instead. Note that `native_namespace` is still available
            (and won't emit a deprecation warning) if you use `narwhals.stable.v1`,
            see [perfect backwards compatibility policy](../backcompat.md/).

    Returns:
        A new Series

    Examples:
        >>> import pandas as pd
        >>> import narwhals as nw
        >>>
        >>> values = [4, 1, 2, 3]
        >>> nw.new_series(name="a", values=values, dtype=nw.Int32, backend=pd)
        ┌─────────────────────┐
        |   Narwhals Series   |
        |---------------------|
        |0    4               |
        |1    1               |
        |2    2               |
        |3    3               |
        |Name: a, dtype: int32|
        └─────────────────────┘
    """
    backend = cast("ModuleType | Implementation | str", backend)
    return _new_series_impl(name, values, dtype, backend=backend, version=Version.MAIN)


def _new_series_impl(
    name: str,
    values: Any,
    dtype: DType | type[DType] | None = None,
    *,
    backend: ModuleType | Implementation | str,
    version: Version,
) -> Series[Any]:
    implementation = Implementation.from_backend(backend)
    if is_eager_allowed(implementation):
        ns = version.namespace.from_backend(implementation).compliant
        series = ns._series.from_iterable(values, name=name, context=ns, dtype=dtype)
        return series.to_narwhals()
    elif implementation is Implementation.DASK:  # pragma: no cover
        msg = "Dask support in Narwhals is lazy-only, so `new_series` is not supported"
        raise NotImplementedError(msg)
    else:  # pragma: no cover
        native_namespace = implementation.to_native_namespace()
        try:
            native_series: NativeSeries = native_namespace.new_series(name, values, dtype)
            return from_native(native_series, series_only=True).alias(name)
        except AttributeError as e:
            msg = "Unknown namespace is expected to implement `new_series` constructor."
            raise AttributeError(msg) from e


@deprecate_native_namespace(warn_version="1.26.0")
def from_dict(
    data: Mapping[str, Any],
    schema: Mapping[str, DType] | Schema | None = None,
    *,
    backend: ModuleType | Implementation | str | None = None,
    native_namespace: ModuleType | None = None,  # noqa: ARG001
) -> DataFrame[Any]:
    """Instantiate DataFrame from dictionary.

    Indexes (if present, for pandas-like backends) are aligned following
    the [left-hand-rule](../concepts/pandas_index.md/).

    Notes:
        For pandas-like dataframes, conversion to schema is applied after dataframe
        creation.

    Arguments:
        data: Dictionary to create DataFrame from.
        schema: The DataFrame schema as Schema or dict of {name: type}. If not
            specified, the schema will be inferred by the native library.
        backend: specifies which eager backend instantiate to. Only
            necessary if inputs are not Narwhals Series.

            `backend` can be specified in various ways

            - As `Implementation.<BACKEND>` with `BACKEND` being `PANDAS`, `PYARROW`,
                `POLARS`, `MODIN` or `CUDF`.
            - As a string: `"pandas"`, `"pyarrow"`, `"polars"`, `"modin"` or `"cudf"`.
            - Directly as a module `pandas`, `pyarrow`, `polars`, `modin` or `cudf`.
        native_namespace: The native library to use for DataFrame creation.

            *Deprecated* (v1.26.0)

            Please use `backend` instead. Note that `native_namespace` is still available
            (and won't emit a deprecation warning) if you use `narwhals.stable.v1`,
            see [perfect backwards compatibility policy](../backcompat.md/).

    Returns:
        A new DataFrame.

    Examples:
        >>> import pandas as pd
        >>> import narwhals as nw
        >>> data = {"c": [5, 2], "d": [1, 4]}
        >>> nw.from_dict(data, backend="pandas")
        ┌──────────────────┐
        |Narwhals DataFrame|
        |------------------|
        |        c  d      |
        |     0  5  1      |
        |     1  2  4      |
        └──────────────────┘
    """
    return _from_dict_impl(data, schema, backend=backend, version=Version.MAIN)


def _from_dict_impl(
    data: Mapping[str, Any],
    schema: Mapping[str, DType] | Schema | None,
    *,
    backend: ModuleType | Implementation | str | None,
    version: Version,
) -> DataFrame[Any]:
    if not data:
        msg = "from_dict cannot be called with empty dictionary"
        raise ValueError(msg)
    if backend is None:
        data, backend = _from_dict_no_backend(data)
    implementation = Implementation.from_backend(backend)
    if is_eager_allowed(implementation):
        ns = version.namespace.from_backend(implementation).compliant
        return ns._dataframe.from_dict(data, schema=schema, context=ns).to_narwhals()
    elif implementation is Implementation.UNKNOWN:  # pragma: no cover
        native_namespace = implementation.to_native_namespace()
        try:
            # implementation is UNKNOWN, Narwhals extension using this feature should
            # implement `from_dict` function in the top-level namespace.
            native_frame: NativeFrame = native_namespace.from_dict(data, schema=schema)
        except AttributeError as e:
            msg = "Unknown namespace is expected to implement `from_dict` function."
            raise AttributeError(msg) from e
        return from_native(native_frame, eager_only=True)
    msg = (
        f"Unsupported `backend` value.\nExpected one of "
        f"{Implementation.POLARS, Implementation.PANDAS, Implementation.PYARROW, Implementation.MODIN, Implementation.CUDF} "
        f"or None, got: {implementation}."
    )
    raise ValueError(msg)


def _from_dict_no_backend(
    data: Mapping[str, Series[Any] | Any], /
) -> tuple[dict[str, Series[Any] | Any], ModuleType]:
    for val in data.values():
        if is_narwhals_series(val):
            native_namespace = val.__native_namespace__()
            break
    else:
        msg = "Calling `from_dict` without `backend` is only supported if all input values are already Narwhals Series"
        raise TypeError(msg)
    data = {key: to_native(value, pass_through=True) for key, value in data.items()}
    return data, native_namespace


@deprecate_native_namespace(warn_version="1.31.0", required=True)
def from_numpy(
    data: _2DArray,
    schema: Mapping[str, DType] | Schema | Sequence[str] | None = None,
    *,
    backend: ModuleType | Implementation | str | None = None,
    native_namespace: ModuleType | None = None,  # noqa: ARG001
) -> DataFrame[Any]:
    """Construct a DataFrame from a NumPy ndarray.

    Notes:
        Only row orientation is currently supported.

        For pandas-like dataframes, conversion to schema is applied after dataframe
        creation.

    Arguments:
        data: Two-dimensional data represented as a NumPy ndarray.
        schema: The DataFrame schema as Schema, dict of {name: type}, or a sequence of str.
        backend: specifies which eager backend instantiate to.

            `backend` can be specified in various ways

            - As `Implementation.<BACKEND>` with `BACKEND` being `PANDAS`, `PYARROW`,
                `POLARS`, `MODIN` or `CUDF`.
            - As a string: `"pandas"`, `"pyarrow"`, `"polars"`, `"modin"` or `"cudf"`.
            - Directly as a module `pandas`, `pyarrow`, `polars`, `modin` or `cudf`.
        native_namespace: The native library to use for DataFrame creation.

            *Deprecated* (v1.31.0)

            Please use `backend` instead. Note that `native_namespace` is still available
            (and won't emit a deprecation warning) if you use `narwhals.stable.v1`,
            see [perfect backwards compatibility policy](../backcompat.md/).

    Returns:
        A new DataFrame.

    Examples:
        >>> import numpy as np
        >>> import pyarrow as pa
        >>> import narwhals as nw
        >>>
        >>> arr = np.array([[5, 2, 1], [1, 4, 3]])
        >>> schema = {"c": nw.Int16(), "d": nw.Float32(), "e": nw.Int8()}
        >>> nw.from_numpy(arr, schema=schema, backend="pyarrow")
        ┌──────────────────┐
        |Narwhals DataFrame|
        |------------------|
        |  pyarrow.Table   |
        |  c: int16        |
        |  d: float        |
        |  e: int8         |
        |  ----            |
        |  c: [[5,1]]      |
        |  d: [[2,4]]      |
        |  e: [[1,3]]      |
        └──────────────────┘
    """
    backend = cast("ModuleType | Implementation | str", backend)
    return _from_numpy_impl(data, schema, backend=backend, version=Version.MAIN)


def _from_numpy_impl(
    data: _2DArray,
    schema: Mapping[str, DType] | Schema | Sequence[str] | None = None,
    *,
    backend: ModuleType | Implementation | str,
    version: Version,
) -> DataFrame[Any]:
    if not is_numpy_array_2d(data):
        msg = "`from_numpy` only accepts 2D numpy arrays"
        raise ValueError(msg)
    if not _is_into_schema(schema):
        msg = (
            "`schema` is expected to be one of the following types: "
            "Mapping[str, DType] | Schema | Sequence[str]. "
            f"Got {type(schema)}."
        )
        raise TypeError(msg)
    implementation = Implementation.from_backend(backend)
    if is_eager_allowed(implementation):
        ns = version.namespace.from_backend(implementation).compliant
        return ns.from_numpy(data, schema).to_narwhals()
    else:  # pragma: no cover
        native_namespace = implementation.to_native_namespace()
        try:
            # implementation is UNKNOWN, Narwhals extension using this feature should
            # implement `from_numpy` function in the top-level namespace.
            native_frame: NativeFrame = native_namespace.from_numpy(data, schema=schema)
        except AttributeError as e:
            msg = "Unknown namespace is expected to implement `from_numpy` function."
            raise AttributeError(msg) from e
        return from_native(native_frame, eager_only=True)


def _is_into_schema(obj: Any) -> TypeIs[_IntoSchema]:
    from narwhals.schema import Schema

    return (
        obj is None or isinstance(obj, (Mapping, Schema)) or is_sequence_but_not_str(obj)
    )


@deprecate_native_namespace(warn_version="1.31.0", required=True)
def from_arrow(
    native_frame: IntoArrowTable,
    *,
    backend: ModuleType | Implementation | str | None = None,
    native_namespace: ModuleType | None = None,  # noqa: ARG001
) -> DataFrame[Any]:  # pragma: no cover
    """Construct a DataFrame from an object which supports the PyCapsule Interface.

    Arguments:
        native_frame: Object which implements `__arrow_c_stream__`.
        backend: specifies which eager backend instantiate to.

            `backend` can be specified in various ways

            - As `Implementation.<BACKEND>` with `BACKEND` being `PANDAS`, `PYARROW`,
                `POLARS`, `MODIN` or `CUDF`.
            - As a string: `"pandas"`, `"pyarrow"`, `"polars"`, `"modin"` or `"cudf"`.
            - Directly as a module `pandas`, `pyarrow`, `polars`, `modin` or `cudf`.
        native_namespace: The native library to use for DataFrame creation.

            *Deprecated* (v1.31.0)

            Please use `backend` instead. Note that `native_namespace` is still available
            (and won't emit a deprecation warning) if you use `narwhals.stable.v1`,
            see [perfect backwards compatibility policy](../backcompat.md/).

    Returns:
        A new DataFrame.

    Examples:
        >>> import pandas as pd
        >>> import polars as pl
        >>> import narwhals as nw
        >>>
        >>> df_native = pd.DataFrame({"a": [1, 2], "b": [4.2, 5.1]})
        >>> nw.from_arrow(df_native, backend="polars")
        ┌──────────────────┐
        |Narwhals DataFrame|
        |------------------|
        |  shape: (2, 2)   |
        |  ┌─────┬─────┐   |
        |  │ a   ┆ b   │   |
        |  │ --- ┆ --- │   |
        |  │ i64 ┆ f64 │   |
        |  ╞═════╪═════╡   |
        |  │ 1   ┆ 4.2 │   |
        |  │ 2   ┆ 5.1 │   |
        |  └─────┴─────┘   |
        └──────────────────┘
    """
    backend = cast("ModuleType | Implementation | str", backend)
    return _from_arrow_impl(native_frame, backend=backend, version=Version.MAIN)


def _from_arrow_impl(
    data: IntoArrowTable,
    *,
    backend: ModuleType | Implementation | str,
    version: Version,
) -> DataFrame[Any]:
    if not (supports_arrow_c_stream(data) or is_pyarrow_table(data)):
        msg = f"Given object of type {type(data)} does not support PyCapsule interface"
        raise TypeError(msg)
    implementation = Implementation.from_backend(backend)
    if is_eager_allowed(implementation):
        ns = version.namespace.from_backend(implementation).compliant
        return ns._dataframe.from_arrow(data, context=ns).to_narwhals()
    else:  # pragma: no cover
        native_namespace = implementation.to_native_namespace()
        try:
            # implementation is UNKNOWN, Narwhals extension using this feature should
            # implement PyCapsule support
            native_frame: NativeFrame = native_namespace.DataFrame(data)
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
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version

    from narwhals import __version__

    deps = ("pandas", "polars", "cudf", "modin", "pyarrow", "numpy")
    deps_info = {"narwhals": __version__}

    for modname in deps:
        try:
            deps_info[modname] = version(modname)
        except PackageNotFoundError:  # noqa: PERF203
            deps_info[modname] = ""
    return deps_info


def show_versions() -> None:
    """Print useful debugging information.

    Examples:
        >>> from narwhals import show_versions
        >>> show_versions()  # doctest: +SKIP
    """
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
        This can be one of

            - 'full': full Narwhals API support
            - 'lazy': only lazy operations are supported. This excludes anything
              which involves iterating over rows in Python.
            - 'interchange': only metadata operations are supported (`df.schema`)
    """
    return obj._level


@deprecate_native_namespace(warn_version="1.27.2", required=True)
def read_csv(
    source: str,
    *,
    backend: ModuleType | Implementation | str | None = None,
    native_namespace: ModuleType | None = None,  # noqa: ARG001
    **kwargs: Any,
) -> DataFrame[Any]:
    """Read a CSV file into a DataFrame.

    Arguments:
        source: Path to a file.
        backend: The eager backend for DataFrame creation.
            `backend` can be specified in various ways

            - As `Implementation.<BACKEND>` with `BACKEND` being `PANDAS`, `PYARROW`,
                `POLARS`, `MODIN` or `CUDF`.
            - As a string: `"pandas"`, `"pyarrow"`, `"polars"`, `"modin"` or `"cudf"`.
            - Directly as a module `pandas`, `pyarrow`, `polars`, `modin` or `cudf`.
        native_namespace: The native library to use for DataFrame creation.

            *Deprecated* (v1.27.2)

            Please use `backend` instead. Note that `native_namespace` is still available
            (and won't emit a deprecation warning) if you use `narwhals.stable.v1`,
            see [perfect backwards compatibility policy](../backcompat.md/).
        kwargs: Extra keyword arguments which are passed to the native CSV reader.
            For example, you could use
            `nw.read_csv('file.csv', backend='pandas', engine='pyarrow')`.

    Returns:
        DataFrame.

    Examples:
        >>> import narwhals as nw
        >>> nw.read_csv("file.csv", backend="pandas")  # doctest:+SKIP
        ┌──────────────────┐
        |Narwhals DataFrame|
        |------------------|
        |        a   b     |
        |     0  1   4     |
        |     1  2   5     |
        └──────────────────┘
    """
    backend = cast("ModuleType | Implementation | str", backend)
    return _read_csv_impl(source, backend=backend, **kwargs)


def _read_csv_impl(
    source: str, *, backend: ModuleType | Implementation | str, **kwargs: Any
) -> DataFrame[Any]:
    eager_backend = Implementation.from_backend(backend)
    native_namespace = eager_backend.to_native_namespace()
    native_frame: NativeFrame
    if eager_backend in {
        Implementation.POLARS,
        Implementation.PANDAS,
        Implementation.MODIN,
        Implementation.CUDF,
    }:
        native_frame = native_namespace.read_csv(source, **kwargs)
    elif eager_backend is Implementation.PYARROW:
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


@deprecate_native_namespace(warn_version="1.31.0", required=True)
def scan_csv(
    source: str,
    *,
    backend: ModuleType | Implementation | str | None = None,
    native_namespace: ModuleType | None = None,  # noqa: ARG001
    **kwargs: Any,
) -> LazyFrame[Any]:
    """Lazily read from a CSV file.

    For the libraries that do not support lazy dataframes, the function reads
    a csv file eagerly and then converts the resulting dataframe to a lazyframe.

    Arguments:
        source: Path to a file.
        backend: The eager backend for DataFrame creation.
            `backend` can be specified in various ways

            - As `Implementation.<BACKEND>` with `BACKEND` being `PANDAS`, `PYARROW`,
                `POLARS`, `MODIN` or `CUDF`.
            - As a string: `"pandas"`, `"pyarrow"`, `"polars"`, `"modin"` or `"cudf"`.
            - Directly as a module `pandas`, `pyarrow`, `polars`, `modin` or `cudf`.
        native_namespace: The native library to use for DataFrame creation.

            *Deprecated* (v1.31.0)

            Please use `backend` instead. Note that `native_namespace` is still available
            (and won't emit a deprecation warning) if you use `narwhals.stable.v1`,
            see [perfect backwards compatibility policy](../backcompat.md/).
        kwargs: Extra keyword arguments which are passed to the native CSV reader.
            For example, you could use
            `nw.scan_csv('file.csv', backend=pd, engine='pyarrow')`.

    Returns:
        LazyFrame.

    Examples:
        >>> import duckdb
        >>> import narwhals as nw
        >>>
        >>> nw.scan_csv("file.csv", backend="duckdb").to_native()  # doctest:+SKIP
        ┌─────────┬───────┐
        │    a    │   b   │
        │ varchar │ int32 │
        ├─────────┼───────┤
        │ x       │     1 │
        │ y       │     2 │
        │ z       │     3 │
        └─────────┴───────┘
    """
    backend = cast("ModuleType | Implementation | str", backend)
    return _scan_csv_impl(source, backend=backend, **kwargs)


def _scan_csv_impl(
    source: str, *, backend: ModuleType | Implementation | str, **kwargs: Any
) -> LazyFrame[Any]:
    implementation = Implementation.from_backend(backend)
    native_namespace = implementation.to_native_namespace()
    native_frame: NativeFrame | NativeLazyFrame
    if implementation is Implementation.POLARS:
        native_frame = native_namespace.scan_csv(source, **kwargs)
    elif implementation in {
        Implementation.PANDAS,
        Implementation.MODIN,
        Implementation.CUDF,
        Implementation.DASK,
        Implementation.DUCKDB,
    }:
        native_frame = native_namespace.read_csv(source, **kwargs)
    elif implementation is Implementation.PYARROW:
        from pyarrow import csv  # ignore-banned-import

        native_frame = csv.read_csv(source, **kwargs)
    elif implementation.is_spark_like():
        if (session := kwargs.pop("session", None)) is None:
            msg = "Spark like backends require a session object to be passed in `kwargs`."
            raise ValueError(msg)

        csv_reader = session.read.format("csv")
        native_frame = (
            csv_reader.load(source)
            if (
                implementation is Implementation.SQLFRAME
                and parse_version(version("sqlframe")) < (3, 27, 0)
            )
            else csv_reader.options(**kwargs).load(source)
        )
    else:  # pragma: no cover
        try:
            # implementation is UNKNOWN, Narwhals extension using this feature should
            # implement `scan_csv` function in the top-level namespace.
            native_frame = native_namespace.scan_csv(source=source, **kwargs)
        except AttributeError as e:
            msg = "Unknown namespace is expected to implement `scan_csv` function."
            raise AttributeError(msg) from e
    return from_native(native_frame).lazy()


@deprecate_native_namespace(warn_version="1.31.0", required=True)
def read_parquet(
    source: str,
    *,
    backend: ModuleType | Implementation | str | None = None,
    native_namespace: ModuleType | None = None,  # noqa: ARG001
    **kwargs: Any,
) -> DataFrame[Any]:
    """Read into a DataFrame from a parquet file.

    Arguments:
        source: Path to a file.
        backend: The eager backend for DataFrame creation.
            `backend` can be specified in various ways

            - As `Implementation.<BACKEND>` with `BACKEND` being `PANDAS`, `PYARROW`,
                `POLARS`, `MODIN` or `CUDF`.
            - As a string: `"pandas"`, `"pyarrow"`, `"polars"`, `"modin"` or `"cudf"`.
            - Directly as a module `pandas`, `pyarrow`, `polars`, `modin` or `cudf`.
        native_namespace: The native library to use for DataFrame creation.

            *Deprecated* (v1.31.0)

            Please use `backend` instead. Note that `native_namespace` is still available
            (and won't emit a deprecation warning) if you use `narwhals.stable.v1`,
            see [perfect backwards compatibility policy](../backcompat.md/).
        kwargs: Extra keyword arguments which are passed to the native parquet reader.
            For example, you could use
            `nw.read_parquet('file.parquet', backend=pd, engine='pyarrow')`.

    Returns:
        DataFrame.

    Examples:
        >>> import pyarrow as pa
        >>> import narwhals as nw
        >>>
        >>> nw.read_parquet("file.parquet", backend="pyarrow")  # doctest:+SKIP
        ┌──────────────────┐
        |Narwhals DataFrame|
        |------------------|
        |pyarrow.Table     |
        |a: int64          |
        |c: double         |
        |----              |
        |a: [[1,2]]        |
        |c: [[0.2,0.1]]    |
        └──────────────────┘
    """
    backend = cast("ModuleType | Implementation | str", backend)
    return _read_parquet_impl(source, backend=backend, **kwargs)


def _read_parquet_impl(
    source: str, *, backend: ModuleType | Implementation | str, **kwargs: Any
) -> DataFrame[Any]:
    implementation = Implementation.from_backend(backend)
    native_namespace = implementation.to_native_namespace()
    native_frame: NativeFrame
    if implementation in {
        Implementation.POLARS,
        Implementation.PANDAS,
        Implementation.MODIN,
        Implementation.CUDF,
        Implementation.DUCKDB,
    }:
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


@deprecate_native_namespace(warn_version="1.31.0", required=True)
def scan_parquet(
    source: str,
    *,
    backend: ModuleType | Implementation | str | None = None,
    native_namespace: ModuleType | None = None,  # noqa: ARG001
    **kwargs: Any,
) -> LazyFrame[Any]:
    """Lazily read from a parquet file.

    For the libraries that do not support lazy dataframes, the function reads
    a parquet file eagerly and then converts the resulting dataframe to a lazyframe.

    Note:
        Spark like backends require a session object to be passed in `kwargs`.

        For instance:

        ```py
        import narwhals as nw
        from sqlframe.duckdb import DuckDBSession

        nw.scan_parquet(source, backend="sqlframe", session=DuckDBSession())
        ```

    Arguments:
        source: Path to a file.
        backend: The eager backend for DataFrame creation.
            `backend` can be specified in various ways

            - As `Implementation.<BACKEND>` with `BACKEND` being `PANDAS`, `PYARROW`,
                `POLARS`, `MODIN`, `CUDF`, `PYSPARK` or `SQLFRAME`.
            - As a string: `"pandas"`, `"pyarrow"`, `"polars"`, `"modin"`, `"cudf"`,
                `"pyspark"` or `"sqlframe"`.
            - Directly as a module `pandas`, `pyarrow`, `polars`, `modin`, `cudf`,
                `pyspark.sql` or `sqlframe`.
        native_namespace: The native library to use for DataFrame creation.

            *Deprecated* (v1.31.0)

            Please use `backend` instead. Note that `native_namespace` is still available
            (and won't emit a deprecation warning) if you use `narwhals.stable.v1`,
            see [perfect backwards compatibility policy](../backcompat.md/).
        kwargs: Extra keyword arguments which are passed to the native parquet reader.
            For example, you could use
            `nw.scan_parquet('file.parquet', backend=pd, engine='pyarrow')`.

    Returns:
        LazyFrame.

    Examples:
        >>> import dask.dataframe as dd
        >>> from sqlframe.duckdb import DuckDBSession
        >>> import narwhals as nw
        >>>
        >>> nw.scan_parquet("file.parquet", backend="dask").collect()  # doctest:+SKIP
        ┌──────────────────┐
        |Narwhals DataFrame|
        |------------------|
        |        a   b     |
        |     0  1   4     |
        |     1  2   5     |
        └──────────────────┘
        >>> nw.scan_parquet(
        ...     "file.parquet", backend="sqlframe", session=DuckDBSession()
        ... ).collect()  # doctest:+SKIP
        ┌──────────────────┐
        |Narwhals DataFrame|
        |------------------|
        |  pyarrow.Table   |
        |  a: int64        |
        |  b: int64        |
        |  ----            |
        |  a: [[1,2]]      |
        |  b: [[4,5]]      |
        └──────────────────┘
    """
    backend = cast("ModuleType | Implementation | str", backend)
    return _scan_parquet_impl(source, backend=backend, **kwargs)


def _scan_parquet_impl(
    source: str, *, backend: ModuleType | Implementation | str, **kwargs: Any
) -> LazyFrame[Any]:
    implementation = Implementation.from_backend(backend)
    native_namespace = implementation.to_native_namespace()
    native_frame: NativeFrame | NativeLazyFrame
    if implementation is Implementation.POLARS:
        native_frame = native_namespace.scan_parquet(source, **kwargs)
    elif implementation in {
        Implementation.PANDAS,
        Implementation.MODIN,
        Implementation.CUDF,
        Implementation.DASK,
        Implementation.DUCKDB,
    }:
        native_frame = native_namespace.read_parquet(source, **kwargs)
    elif implementation is Implementation.PYARROW:
        import pyarrow.parquet as pq  # ignore-banned-import

        native_frame = pq.read_table(source, **kwargs)
    elif implementation.is_spark_like():
        if (session := kwargs.pop("session", None)) is None:
            msg = "Spark like backends require a session object to be passed in `kwargs`."
            raise ValueError(msg)

        pq_reader = session.read.format("parquet")
        native_frame = (
            pq_reader.load(source)
            if (
                implementation is Implementation.SQLFRAME
                and parse_version(version("sqlframe")) < (3, 27, 0)
            )
            else pq_reader.options(**kwargs).load(source)
        )

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

    Examples:
        >>> import polars as pl
        >>> import narwhals as nw
        >>>
        >>> df_native = pl.DataFrame({"a": [1, 2], "b": [3, 4], "c": ["x", "z"]})
        >>> nw.from_native(df_native).select(nw.col("a", "b") * nw.col("b"))
        ┌──────────────────┐
        |Narwhals DataFrame|
        |------------------|
        |  shape: (2, 2)   |
        |  ┌─────┬─────┐   |
        |  │ a   ┆ b   │   |
        |  │ --- ┆ --- │   |
        |  │ i64 ┆ i64 │   |
        |  ╞═════╪═════╡   |
        |  │ 3   ┆ 9   │   |
        |  │ 8   ┆ 16  │   |
        |  └─────┴─────┘   |
        └──────────────────┘
    """
    flat_names = flatten(names)

    def func(plx: Any) -> Any:
        return plx.col(*flat_names)

    return Expr(
        func,
        ExprMetadata.selector_single()
        if len(flat_names) == 1
        else ExprMetadata.selector_multi_named(),
    )


def exclude(*names: str | Iterable[str]) -> Expr:
    """Creates an expression that excludes columns by their name(s).

    Arguments:
        names: Name(s) of the columns to exclude.

    Returns:
        A new expression.

    Examples:
        >>> import polars as pl
        >>> import narwhals as nw
        >>>
        >>> df_native = pl.DataFrame({"a": [1, 2], "b": [3, 4], "c": ["x", "z"]})
        >>> nw.from_native(df_native).select(nw.exclude("c", "a"))
        ┌──────────────────┐
        |Narwhals DataFrame|
        |------------------|
        |  shape: (2, 1)   |
        |  ┌─────┐         |
        |  │ b   │         |
        |  │ --- │         |
        |  │ i64 │         |
        |  ╞═════╡         |
        |  │ 3   │         |
        |  │ 4   │         |
        |  └─────┘         |
        └──────────────────┘
    """
    exclude_names = frozenset(flatten(names))

    def func(plx: Any) -> Any:
        return plx.exclude(exclude_names)

    return Expr(func, ExprMetadata.selector_multi_unnamed())


def nth(*indices: int | Sequence[int]) -> Expr:
    """Creates an expression that references one or more columns by their index(es).

    Notes:
        `nth` is not supported for Polars version<1.0.0. Please use
        [`narwhals.col`][] instead.

    Arguments:
        indices: One or more indices representing the columns to retrieve.

    Returns:
        A new expression.

    Examples:
        >>> import pyarrow as pa
        >>> import narwhals as nw
        >>>
        >>> df_native = pa.table({"a": [1, 2], "b": [3, 4], "c": [0.123, 3.14]})
        >>> nw.from_native(df_native).select(nw.nth(0, 2) * 2)
        ┌──────────────────┐
        |Narwhals DataFrame|
        |------------------|
        |pyarrow.Table     |
        |a: int64          |
        |c: double         |
        |----              |
        |a: [[2,4]]        |
        |c: [[0.246,6.28]] |
        └──────────────────┘
    """
    flat_indices = flatten(indices)

    def func(plx: Any) -> Any:
        return plx.nth(*flat_indices)

    return Expr(
        func,
        ExprMetadata.selector_single()
        if len(flat_indices) == 1
        else ExprMetadata.selector_multi_unnamed(),
    )


# Add underscore so it doesn't conflict with builtin `all`
def all_() -> Expr:
    """Instantiate an expression representing all columns.

    Returns:
        A new expression.

    Examples:
        >>> import pandas as pd
        >>> import narwhals as nw
        >>>
        >>> df_native = pd.DataFrame({"a": [1, 2], "b": [3.14, 0.123]})
        >>> nw.from_native(df_native).select(nw.all() * 2)
        ┌──────────────────┐
        |Narwhals DataFrame|
        |------------------|
        |      a      b    |
        |   0  2  6.280    |
        |   1  4  0.246    |
        └──────────────────┘
    """
    return Expr(lambda plx: plx.all(), ExprMetadata.selector_multi_unnamed())


# Add underscore so it doesn't conflict with builtin `len`
def len_() -> Expr:
    """Return the number of rows.

    Returns:
        A new expression.

    Examples:
        >>> import polars as pl
        >>> import narwhals as nw
        >>>
        >>> df_native = pl.DataFrame({"a": [1, 2], "b": [5, None]})
        >>> nw.from_native(df_native).select(nw.len())
        ┌──────────────────┐
        |Narwhals DataFrame|
        |------------------|
        |  shape: (1, 1)   |
        |  ┌─────┐         |
        |  │ len │         |
        |  │ --- │         |
        |  │ u32 │         |
        |  ╞═════╡         |
        |  │ 2   │         |
        |  └─────┘         |
        └──────────────────┘
    """

    def func(plx: Any) -> Any:
        return plx.len()

    return Expr(func, ExprMetadata.aggregation())


def sum(*columns: str) -> Expr:
    """Sum all values.

    Note:
        Syntactic sugar for ``nw.col(columns).sum()``

    Arguments:
        columns: Name(s) of the columns to use in the aggregation function

    Returns:
        A new expression.

    Examples:
        >>> import pandas as pd
        >>> import narwhals as nw
        >>>
        >>> df_native = pd.DataFrame({"a": [1, 2], "b": [-1.4, 6.2]})
        >>> nw.from_native(df_native).select(nw.sum("a", "b"))
        ┌──────────────────┐
        |Narwhals DataFrame|
        |------------------|
        |       a    b     |
        |    0  3  4.8     |
        └──────────────────┘
    """
    return col(*columns).sum()


def mean(*columns: str) -> Expr:
    """Get the mean value.

    Note:
        Syntactic sugar for ``nw.col(columns).mean()``

    Arguments:
        columns: Name(s) of the columns to use in the aggregation function

    Returns:
        A new expression.

    Examples:
        >>> import pyarrow as pa
        >>> import narwhals as nw
        >>>
        >>> df_native = pa.table({"a": [1, 8, 3], "b": [3.14, 6.28, 42.1]})
        >>> nw.from_native(df_native).select(nw.mean("a", "b"))
        ┌─────────────────────────┐
        |   Narwhals DataFrame    |
        |-------------------------|
        |pyarrow.Table            |
        |a: double                |
        |b: double                |
        |----                     |
        |a: [[4]]                 |
        |b: [[17.173333333333336]]|
        └─────────────────────────┘
    """
    return col(*columns).mean()


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

    Examples:
        >>> import polars as pl
        >>> import narwhals as nw
        >>>
        >>> df_native = pl.DataFrame({"a": [4, 5, 2]})
        >>> nw.from_native(df_native).select(nw.median("a"))
        ┌──────────────────┐
        |Narwhals DataFrame|
        |------------------|
        |  shape: (1, 1)   |
        |  ┌─────┐         |
        |  │ a   │         |
        |  │ --- │         |
        |  │ f64 │         |
        |  ╞═════╡         |
        |  │ 4.0 │         |
        |  └─────┘         |
        └──────────────────┘
    """
    return col(*columns).median()


def min(*columns: str) -> Expr:
    """Return the minimum value.

    Note:
       Syntactic sugar for ``nw.col(columns).min()``.

    Arguments:
        columns: Name(s) of the columns to use in the aggregation function.

    Returns:
        A new expression.

    Examples:
        >>> import pyarrow as pa
        >>> import narwhals as nw
        >>>
        >>> df_native = pa.table({"a": [1, 2], "b": [5, 10]})
        >>> nw.from_native(df_native).select(nw.min("a", "b"))
        ┌──────────────────┐
        |Narwhals DataFrame|
        |------------------|
        |  pyarrow.Table   |
        |  a: int64        |
        |  b: int64        |
        |  ----            |
        |  a: [[1]]        |
        |  b: [[5]]        |
        └──────────────────┘
    """
    return col(*columns).min()


def max(*columns: str) -> Expr:
    """Return the maximum value.

    Note:
       Syntactic sugar for ``nw.col(columns).max()``.

    Arguments:
        columns: Name(s) of the columns to use in the aggregation function.

    Returns:
        A new expression.

    Examples:
        >>> import pandas as pd
        >>> import narwhals as nw
        >>>
        >>> df_native = pd.DataFrame({"a": [1, 2], "b": [5, 10]})
        >>> nw.from_native(df_native).select(nw.max("a", "b"))
        ┌──────────────────┐
        |Narwhals DataFrame|
        |------------------|
        |        a   b     |
        |     0  2  10     |
        └──────────────────┘
    """
    return col(*columns).max()


def sum_horizontal(*exprs: IntoExpr | Iterable[IntoExpr]) -> Expr:
    """Sum all values horizontally across columns.

    Warning:
        Unlike Polars, we support horizontal sum over numeric columns only.

    Arguments:
        exprs: Name(s) of the columns to use in the aggregation function. Accepts
            expression input.

    Returns:
        A new expression.

    Examples:
        >>> import polars as pl
        >>> import narwhals as nw
        >>>
        >>> df_native = pl.DataFrame({"a": [1, 2, 3], "b": [5, 10, None]})
        >>> nw.from_native(df_native).with_columns(sum=nw.sum_horizontal("a", "b"))
        ┌────────────────────┐
        | Narwhals DataFrame |
        |--------------------|
        |shape: (3, 3)       |
        |┌─────┬──────┬─────┐|
        |│ a   ┆ b    ┆ sum │|
        |│ --- ┆ ---  ┆ --- │|
        |│ i64 ┆ i64  ┆ i64 │|
        |╞═════╪══════╪═════╡|
        |│ 1   ┆ 5    ┆ 6   │|
        |│ 2   ┆ 10   ┆ 12  │|
        |│ 3   ┆ null ┆ 3   │|
        |└─────┴──────┴─────┘|
        └────────────────────┘
    """
    if not exprs:
        msg = "At least one expression must be passed to `sum_horizontal`"
        raise ValueError(msg)
    flat_exprs = flatten(exprs)
    return Expr(
        lambda plx: apply_n_ary_operation(
            plx, plx.sum_horizontal, *flat_exprs, str_as_lit=False
        ),
        ExprMetadata.from_horizontal_op(*flat_exprs),
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

    Examples:
        >>> import pyarrow as pa
        >>> import narwhals as nw
        >>>
        >>> df_native = pa.table({"a": [1, 8, 3], "b": [4, 5, None]})
        >>> nw.from_native(df_native).with_columns(h_min=nw.min_horizontal("a", "b"))
        ┌──────────────────┐
        |Narwhals DataFrame|
        |------------------|
        | pyarrow.Table    |
        | a: int64         |
        | b: int64         |
        | h_min: int64     |
        | ----             |
        | a: [[1,8,3]]     |
        | b: [[4,5,null]]  |
        | h_min: [[1,5,3]] |
        └──────────────────┘
    """
    if not exprs:
        msg = "At least one expression must be passed to `min_horizontal`"
        raise ValueError(msg)
    flat_exprs = flatten(exprs)
    return Expr(
        lambda plx: apply_n_ary_operation(
            plx, plx.min_horizontal, *flat_exprs, str_as_lit=False
        ),
        ExprMetadata.from_horizontal_op(*flat_exprs),
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

    Examples:
        >>> import polars as pl
        >>> import narwhals as nw
        >>>
        >>> df_native = pl.DataFrame({"a": [1, 8, 3], "b": [4, 5, None]})
        >>> nw.from_native(df_native).with_columns(h_max=nw.max_horizontal("a", "b"))
        ┌──────────────────────┐
        |  Narwhals DataFrame  |
        |----------------------|
        |shape: (3, 3)         |
        |┌─────┬──────┬───────┐|
        |│ a   ┆ b    ┆ h_max │|
        |│ --- ┆ ---  ┆ ---   │|
        |│ i64 ┆ i64  ┆ i64   │|
        |╞═════╪══════╪═══════╡|
        |│ 1   ┆ 4    ┆ 4     │|
        |│ 8   ┆ 5    ┆ 8     │|
        |│ 3   ┆ null ┆ 3     │|
        |└─────┴──────┴───────┘|
        └──────────────────────┘
    """
    if not exprs:
        msg = "At least one expression must be passed to `max_horizontal`"
        raise ValueError(msg)
    flat_exprs = flatten(exprs)
    return Expr(
        lambda plx: apply_n_ary_operation(
            plx, plx.max_horizontal, *flat_exprs, str_as_lit=False
        ),
        ExprMetadata.from_horizontal_op(*flat_exprs),
    )


class When:
    def __init__(self, *predicates: IntoExpr | Iterable[IntoExpr]) -> None:
        self._predicate = all_horizontal(*flatten(predicates))
        check_expressions_preserve_length(self._predicate, function_name="when")

    def then(self, value: IntoExpr | NonNestedLiteral | _1DArray) -> Then:
        return Then(
            lambda plx: apply_n_ary_operation(
                plx,
                lambda *args: plx.when(args[0]).then(args[1]),
                self._predicate,
                value,
                str_as_lit=False,
            ),
            combine_metadata(
                self._predicate,
                value,
                str_as_lit=False,
                allow_multi_output=False,
                to_single_output=False,
            ),
        )


class Then(Expr):
    def otherwise(self, value: IntoExpr | NonNestedLiteral | _1DArray) -> Expr:
        kind = ExprKind.from_into_expr(value, str_as_lit=False)

        def func(plx: CompliantNamespace[Any, Any]) -> CompliantExpr[Any, Any]:
            compliant_expr = self._to_compliant_expr(plx)
            compliant_value = extract_compliant(plx, value, str_as_lit=False)
            if is_scalar_like(kind) and is_compliant_expr(compliant_value):
                compliant_value = compliant_value.broadcast(kind)
            return compliant_expr.otherwise(compliant_value)  # type: ignore[attr-defined, no-any-return]

        return Expr(
            func,
            combine_metadata(
                self,
                value,
                str_as_lit=False,
                allow_multi_output=False,
                to_single_output=False,
            ),
        )


def when(*predicates: IntoExpr | Iterable[IntoExpr]) -> When:
    """Start a `when-then-otherwise` expression.

    Expression similar to an `if-else` statement in Python. Always initiated by a
    `pl.when(<condition>).then(<value if condition>)`, and optionally followed by a
    `.otherwise(<value if condition is false>)` can be appended at the end. If not
    appended, and the condition is not `True`, `None` will be returned.

    Info:
        Chaining multiple `.when(<condition>).then(<value>)` statements is currently
        not supported.
        See [Narwhals#668](https://github.com/narwhals-dev/narwhals/issues/668).

    Arguments:
        predicates: Condition(s) that must be met in order to apply the subsequent
            statement. Accepts one or more boolean expressions, which are implicitly
            combined with `&`. String input is parsed as a column name.

    Returns:
        A "when" object, which `.then` can be called on.

    Examples:
        >>> import pandas as pd
        >>> import narwhals as nw
        >>>
        >>> data = {"a": [1, 2, 3], "b": [5, 10, 15]}
        >>> df_native = pd.DataFrame(data)
        >>> nw.from_native(df_native).with_columns(
        ...     nw.when(nw.col("a") < 3).then(5).otherwise(6).alias("a_when")
        ... )
        ┌──────────────────┐
        |Narwhals DataFrame|
        |------------------|
        |    a   b  a_when |
        | 0  1   5       5 |
        | 1  2  10       5 |
        | 2  3  15       6 |
        └──────────────────┘
    """
    return When(*predicates)


def all_horizontal(*exprs: IntoExpr | Iterable[IntoExpr]) -> Expr:
    r"""Compute the bitwise AND horizontally across columns.

    Arguments:
        exprs: Name(s) of the columns to use in the aggregation function. Accepts
            expression input.

    Returns:
        A new expression.

    Examples:
        >>> import pyarrow as pa
        >>> import narwhals as nw
        >>>
        >>> data = {
        ...     "a": [False, False, True, True, False, None],
        ...     "b": [False, True, True, None, None, None],
        ... }
        >>> df_native = pa.table(data)
        >>> nw.from_native(df_native).select("a", "b", all=nw.all_horizontal("a", "b"))
        ┌─────────────────────────────────────────┐
        |           Narwhals DataFrame            |
        |-----------------------------------------|
        |pyarrow.Table                            |
        |a: bool                                  |
        |b: bool                                  |
        |all: bool                                |
        |----                                     |
        |a: [[false,false,true,true,false,null]]  |
        |b: [[false,true,true,null,null,null]]    |
        |all: [[false,false,true,null,false,null]]|
        └─────────────────────────────────────────┘

    """
    if not exprs:
        msg = "At least one expression must be passed to `all_horizontal`"
        raise ValueError(msg)
    flat_exprs = flatten(exprs)
    return Expr(
        lambda plx: apply_n_ary_operation(
            plx, plx.all_horizontal, *flat_exprs, str_as_lit=False
        ),
        ExprMetadata.from_horizontal_op(*flat_exprs),
    )


def lit(value: NonNestedLiteral, dtype: DType | type[DType] | None = None) -> Expr:
    """Return an expression representing a literal value.

    Arguments:
        value: The value to use as literal.
        dtype: The data type of the literal value. If not provided, the data type will
            be inferred by the native library.

    Returns:
        A new expression.

    Examples:
        >>> import pandas as pd
        >>> import narwhals as nw
        >>>
        >>> df_native = pd.DataFrame({"a": [1, 2]})
        >>> nw.from_native(df_native).with_columns(nw.lit(3))
        ┌──────────────────┐
        |Narwhals DataFrame|
        |------------------|
        |     a  literal   |
        |  0  1        3   |
        |  1  2        3   |
        └──────────────────┘
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

    return Expr(lambda plx: plx.lit(value, dtype), ExprMetadata.literal())


def any_horizontal(*exprs: IntoExpr | Iterable[IntoExpr]) -> Expr:
    r"""Compute the bitwise OR horizontally across columns.

    Arguments:
        exprs: Name(s) of the columns to use in the aggregation function. Accepts
            expression input.

    Returns:
        A new expression.

    Examples:
        >>> import polars as pl
        >>> import narwhals as nw
        >>>
        >>> data = {
        ...     "a": [False, False, True, True, False, None],
        ...     "b": [False, True, True, None, None, None],
        ... }
        >>> df_native = pl.DataFrame(data)
        >>> nw.from_native(df_native).select("a", "b", any=nw.any_horizontal("a", "b"))
        ┌─────────────────────────┐
        |   Narwhals DataFrame    |
        |-------------------------|
        |shape: (6, 3)            |
        |┌───────┬───────┬───────┐|
        |│ a     ┆ b     ┆ any   │|
        |│ ---   ┆ ---   ┆ ---   │|
        |│ bool  ┆ bool  ┆ bool  │|
        |╞═══════╪═══════╪═══════╡|
        |│ false ┆ false ┆ false │|
        |│ false ┆ true  ┆ true  │|
        |│ true  ┆ true  ┆ true  │|
        |│ true  ┆ null  ┆ true  │|
        |│ false ┆ null  ┆ null  │|
        |│ null  ┆ null  ┆ null  │|
        |└───────┴───────┴───────┘|
        └─────────────────────────┘
    """
    if not exprs:
        msg = "At least one expression must be passed to `any_horizontal`"
        raise ValueError(msg)
    flat_exprs = flatten(exprs)
    return Expr(
        lambda plx: apply_n_ary_operation(
            plx, plx.any_horizontal, *flat_exprs, str_as_lit=False
        ),
        ExprMetadata.from_horizontal_op(*flat_exprs),
    )


def mean_horizontal(*exprs: IntoExpr | Iterable[IntoExpr]) -> Expr:
    """Compute the mean of all values horizontally across columns.

    Arguments:
        exprs: Name(s) of the columns to use in the aggregation function. Accepts
            expression input.

    Returns:
        A new expression.

    Examples:
        >>> import pyarrow as pa
        >>> import narwhals as nw
        >>>
        >>> data = {
        ...     "a": [1, 8, 3],
        ...     "b": [4, 5, None],
        ...     "c": ["x", "y", "z"],
        ... }
        >>> df_native = pa.table(data)

        We define a dataframe-agnostic function that computes the horizontal mean of "a"
        and "b" columns:

        >>> nw.from_native(df_native).select(nw.mean_horizontal("a", "b"))
        ┌──────────────────┐
        |Narwhals DataFrame|
        |------------------|
        | pyarrow.Table    |
        | a: double        |
        | ----             |
        | a: [[2.5,6.5,3]] |
        └──────────────────┘
    """
    if not exprs:
        msg = "At least one expression must be passed to `mean_horizontal`"
        raise ValueError(msg)
    flat_exprs = flatten(exprs)
    return Expr(
        lambda plx: apply_n_ary_operation(
            plx, plx.mean_horizontal, *flat_exprs, str_as_lit=False
        ),
        ExprMetadata.from_horizontal_op(*flat_exprs),
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

    Examples:
        >>> import pandas as pd
        >>> import narwhals as nw
        >>>
        >>> data = {
        ...     "a": [1, 2, 3],
        ...     "b": ["dogs", "cats", None],
        ...     "c": ["play", "swim", "walk"],
        ... }
        >>> df_native = pd.DataFrame(data)
        >>> (
        ...     nw.from_native(df_native).select(
        ...         nw.concat_str(
        ...             [
        ...                 nw.col("a") * 2,
        ...                 nw.col("b"),
        ...                 nw.col("c"),
        ...             ],
        ...             separator=" ",
        ...         ).alias("full_sentence")
        ...     )
        ... )
        ┌──────────────────┐
        |Narwhals DataFrame|
        |------------------|
        |   full_sentence  |
        | 0   2 dogs play  |
        | 1   4 cats swim  |
        | 2          None  |
        └──────────────────┘
    """
    flat_exprs = flatten([*flatten([exprs]), *more_exprs])
    return Expr(
        lambda plx: apply_n_ary_operation(
            plx,
            lambda *args: plx.concat_str(
                *args, separator=separator, ignore_nulls=ignore_nulls
            ),
            *flat_exprs,
            str_as_lit=False,
        ),
        combine_metadata(
            *flat_exprs, str_as_lit=False, allow_multi_output=True, to_single_output=True
        ),
    )
