from __future__ import annotations

import secrets
from enum import Enum
from enum import auto
from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable
from typing import TypeVar

from narwhals.dependencies import get_cudf
from narwhals.dependencies import get_modin
from narwhals.dependencies import get_pandas
from narwhals.utils import isinstance_or_issubclass

T = TypeVar("T")

if TYPE_CHECKING:
    from narwhals._pandas_like.expr import PandasExpr
    from narwhals._pandas_like.series import PandasSeries
    from narwhals.dtypes import DType

    ExprT = TypeVar("ExprT", bound=PandasExpr)


class Implementation(Enum):
    PANDAS = auto()
    MODIN = auto()
    CUDF = auto()


def validate_column_comparand(index: Any, other: Any) -> Any:
    """Validate RHS of binary operation.

    If the comparison isn't supported, return `NotImplemented` so that the
    "right-hand-side" operation (e.g. `__radd__`) can be tried.

    If RHS is length 1, return the scalar value, so that the underlying
    library can broadcast it.
    """
    from narwhals._pandas_like.dataframe import PandasDataFrame
    from narwhals._pandas_like.series import PandasSeries

    if isinstance(other, list):
        if len(other) > 1:
            # e.g. `plx.all() + plx.all()`
            msg = "Multi-output expressions are not supported in this context"
            raise ValueError(msg)
        other = other[0]
    if isinstance(other, PandasDataFrame):
        return NotImplemented
    if isinstance(other, PandasSeries):
        if other.len() == 1:
            # broadcast
            return other.item()
        if other._native_series.index is not index:
            return set_axis(
                other._native_series,
                index,
                implementation=other._implementation,
                backend_version=other._backend_version,
            )
        return other._native_series
    return other


def validate_dataframe_comparand(index: Any, other: Any) -> Any:
    """Validate RHS of binary operation.

    If the comparison isn't supported, return `NotImplemented` so that the
    "right-hand-side" operation (e.g. `__radd__`) can be tried.
    """
    from narwhals._pandas_like.dataframe import PandasDataFrame
    from narwhals._pandas_like.series import PandasSeries

    if isinstance(other, PandasDataFrame):
        return NotImplemented
    if isinstance(other, PandasSeries):
        if other.len() == 1:
            # broadcast
            return other._native_series.iloc[0]
        if other._native_series.index is not index:
            return set_axis(
                other._native_series,
                index,
                implementation=other._implementation,
                backend_version=other._backend_version,
            )
        return other._native_series
    raise AssertionError("Please report a bug")


def create_native_series(
    iterable: Any,
    index: Any = None,
    *,
    implementation: Implementation,
    backend_version: tuple[int, ...],
) -> PandasSeries:
    from narwhals._pandas_like.series import PandasSeries

    if implementation is Implementation.PANDAS:
        pd = get_pandas()
        series = pd.Series(iterable, index=index, name="")
    elif implementation is Implementation.MODIN:
        mpd = get_modin()
        series = mpd.Series(iterable, index=index, name="")
    elif implementation is Implementation.CUDF:
        cudf = get_cudf()
        series = cudf.Series(iterable, index=index, name="")
    return PandasSeries(
        series, implementation=implementation, backend_version=backend_version
    )


def is_simple_aggregation(expr: PandasExpr) -> bool:
    """
    Check if expr is a very simple one, such as:

    - nw.col('a').mean()  # depth 1
    - nw.mean('a')  # depth 1
    - nw.len()  # depth 0

    as opposed to, say

    - nw.col('a').filter(nw.col('b')>nw.col('c')).max()

    because then, we can use a fastpath in pandas.
    """
    return expr._depth < 2


def horizontal_concat(
    dfs: list[Any], *, implementation: Implementation, backend_version: tuple[int, ...]
) -> Any:
    """
    Concatenate (native) DataFrames horizontally.

    Should be in namespace.
    """
    if implementation is Implementation.PANDAS:
        pd = get_pandas()

        if backend_version < (3,):
            return pd.concat(dfs, axis=1, copy=False)
        return pd.concat(dfs, axis=1)  # pragma: no cover
    if implementation is Implementation.CUDF:  # pragma: no cover
        cudf = get_cudf()

        return cudf.concat(dfs, axis=1)
    if implementation is Implementation.MODIN:  # pragma: no cover
        mpd = get_modin()

        return mpd.concat(dfs, axis=1)
    msg = f"Unknown implementation: {implementation}"  # pragma: no cover
    raise TypeError(msg)  # pragma: no cover


def vertical_concat(
    dfs: list[Any], *, implementation: Implementation, backend_version: tuple[int, ...]
) -> Any:
    """
    Concatenate (native) DataFrames vertically.

    Should be in namespace.
    """
    if not dfs:
        msg = "No dataframes to concatenate"  # pragma: no cover
        raise AssertionError(msg)
    cols = set(dfs[0].columns)
    for df in dfs:
        cols_current = set(df.columns)
        if cols_current != cols:
            msg = "unable to vstack, column names don't match"
            raise TypeError(msg)
    if implementation is Implementation.PANDAS:
        pd = get_pandas()

        if backend_version < (3,):
            return pd.concat(dfs, axis=0, copy=False)
        return pd.concat(dfs, axis=0)  # pragma: no cover
    if implementation is Implementation.CUDF:  # pragma: no cover
        cudf = get_cudf()

        return cudf.concat(dfs, axis=0)
    if implementation is Implementation.MODIN:  # pragma: no cover
        mpd = get_modin()

        return mpd.concat(dfs, axis=0)
    msg = f"Unknown implementation: {implementation}"  # pragma: no cover
    raise TypeError(msg)  # pragma: no cover


def native_series_from_iterable(
    data: Iterable[Any],
    name: str,
    index: Any,
    implementation: Implementation,
) -> Any:
    """Return native series."""
    if implementation is Implementation.PANDAS:
        pd = get_pandas()

        return pd.Series(data, name=name, index=index, copy=False)
    if implementation is Implementation.CUDF:  # pragma: no cover
        cudf = get_cudf()

        return cudf.Series(data, name=name, index=index)
    if implementation is Implementation.MODIN:  # pragma: no cover
        mpd = get_modin()

        return mpd.Series(data, name=name, index=index)
    msg = f"Unknown implementation: {implementation}"  # pragma: no cover
    raise TypeError(msg)  # pragma: no cover


def set_axis(
    obj: T,
    index: Any,
    *,
    implementation: Implementation,
    backend_version: tuple[int, ...],
) -> T:
    if implementation is Implementation.PANDAS and backend_version < (
        1,
    ):  # pragma: no cover
        kwargs = {"inplace": False}
    else:
        kwargs = {}
    if implementation is Implementation.PANDAS and backend_version >= (
        1,
        5,
    ):  # pragma: no cover
        kwargs["copy"] = False
    else:  # pragma: no cover
        pass
    return obj.set_axis(index, axis=0, **kwargs)  # type: ignore[no-any-return, attr-defined]


def __translate_primitive_dtype(dtype: str) -> DType | None:
    from narwhals import dtypes

    dtype_mappers: dict[str, type[dtypes.DType]] = {
        "int64": dtypes.Int64,
        "int32": dtypes.Int32,
        "int16": dtypes.Int16,
        "int8": dtypes.Int8,
        "uint64": dtypes.UInt64,
        "uint32": dtypes.UInt32,
        "uint16": dtypes.UInt16,
        "uint8": dtypes.UInt8,
        "float64": dtypes.Float64,
        "double": dtypes.Float64,
        "float32": dtypes.Float32,
        "float": dtypes.Float32,
        "string": dtypes.String,
        "string[python]": dtypes.String,
        "large_string": dtypes.String,
        "bool": dtypes.Boolean,
        "boolean": dtypes.Boolean,
        "category": dtypes.Categorical,
        "date32[day]": dtypes.Date,
    }

    if dtype not in dtype_mappers:
        return None

    dtype_factory = next(
        dtype_factory
        for pandas_dtype, dtype_factory in dtype_mappers.items()
        if dtype == pandas_dtype
    )
    return dtype_factory()


def __translate_datetime_dtype(dtype: str) -> DType | None:
    from narwhals import dtypes

    if dtype.startswith("datetime64"):
        # todo: different time units and time zones
        return dtypes.Datetime()
    if dtype.startswith(("timedelta64", "duration")):
        # todo: different time units
        return dtypes.Duration()
    if dtype.startswith("timestamp["):
        # pyarrow-backed datetime
        # todo: different time units and time zones
        return dtypes.Datetime()
    if dtype == "date32[day][pyarrow]":
        return dtypes.Date()

    return None


def __translate_categorical_dtype(dtype: str) -> DType | None:
    from narwhals import dtypes

    if dtype in ("category",) or dtype.startswith("dictionary<"):
        return dtypes.Categorical()

    return None


def __translate_object_dtype(dtype: str, column: Any) -> DType | None:
    from narwhals import dtypes

    if str(dtype) != "object":
        return None

    if (idx := column.first_valid_index()) is not None and isinstance(
        column.loc[idx], str
    ):
        # Infer based on first non-missing value.
        # For pandas pre 3.0, this isn't perfect.
        # After pandas 3.0, pandas has a dedicated string dtype
        # which is inferred by default.
        return dtypes.String()
    return dtypes.Object()


def translate_dtype(column: Any) -> DType:
    from narwhals import dtypes

    dtype = str(column.dtype).lower().replace("[pyarrow]", "")

    primitive_dtype = __translate_primitive_dtype(dtype)
    if primitive_dtype:
        return primitive_dtype

    categorical_dtype = __translate_categorical_dtype(dtype)
    if categorical_dtype:
        return categorical_dtype

    datetime_dtype = __translate_datetime_dtype(dtype)
    if datetime_dtype:
        return datetime_dtype

    object_dtype = __translate_object_dtype(dtype, column)
    if object_dtype:
        return object_dtype

    return dtypes.Unknown()


def get_dtype_backend(dtype: Any, implementation: Implementation) -> str:
    if implementation is Implementation.PANDAS:
        pd = get_pandas()
        if hasattr(pd, "ArrowDtype") and isinstance(dtype, pd.ArrowDtype):
            return "pyarrow-nullable"

        try:
            if isinstance(dtype, pd.core.dtypes.dtypes.BaseMaskedDtype):
                return "pandas-nullable"
        except AttributeError:  # pragma: no cover
            # defensive check for old pandas versions
            pass
        return "numpy"
    else:  # pragma: no cover
        return "numpy"


def reverse_translate_dtype(
    dtype: DType | type[DType], starting_dtype: Any, implementation: Implementation
) -> Any:
    from narwhals import dtypes

    dtypes_mapping: dict[type[dtypes.DType], dict[str, str | Any]] = {
        dtypes.Float64: {
            "pyarrow-nullable": "Float64[pyarrow]",
            "pandas-nullable": "Float64",
            "numpy": "float64",
        },
        dtypes.Float32: {
            "pyarrow-nullable": "Float32[pyarrow]",
            "pandas-nullable": "Float32",
            "numpy": "float32",
        },
        dtypes.Int64: {
            "pyarrow-nullable": "Int64[pyarrow]",
            "pandas-nullable": "Int64",
            "numpy": "int64",
        },
        dtypes.Int32: {
            "pyarrow-nullable": "Int32[pyarrow]",
            "pandas-nullable": "Int32",
            "numpy": "int32",
        },
        dtypes.Int16: {
            "pyarrow-nullable": "Int16[pyarrow]",
            "pandas-nullable": "Int16",
            "numpy": "int16",
        },
        dtypes.Int8: {
            "pyarrow-nullable": "Int8[pyarrow]",
            "pandas-nullable": "Int8",
            "numpy": "int8",
        },
        dtypes.UInt64: {
            "pyarrow-nullable": "UInt64[pyarrow]",
            "pandas-nullable": "UInt64",
            "numpy": "uint64",
        },
        dtypes.UInt32: {
            "pyarrow-nullable": "UInt32[pyarrow]",
            "pandas-nullable": "UInt32",
            "numpy": "uint32",
        },
        dtypes.UInt16: {
            "pyarrow-nullable": "UInt16[pyarrow]",
            "pandas-nullable": "UInt16",
            "numpy": "uint16",
        },
        dtypes.UInt8: {
            "pyarrow-nullable": "UInt8[pyarrow]",
            "pandas-nullable": "UInt8",
            "numpy": "uint8",
        },
        dtypes.String: {
            "pyarrow-nullable": "string[pyarrow]",
            "pandas-nullable": "string",
            "numpy": str,
        },
        dtypes.Boolean: {
            "pyarrow-nullable": "boolean[pyarrow]",
            "pandas-nullable": "boolean",
            "numpy": "bool",
        },
        dtypes.Categorical: {
            "pyarrow-nullable": "category",
            "pandas-nullable": "category",
            "numpy": "category",
        },
        dtypes.Datetime: {
            "pyarrow-nullable": "timestamp[ns][pyarrow]",
            "pandas-nullable": "datetime64[ns]",
            "numpy": "datetime64[ns]",
        },
        dtypes.Duration: {
            "pyarrow-nullable": "duration[ns][pyarrow]",
            "pandas-nullable": "timedelta64[ns]",
            "numpy": "timedelta64[ns]",
        },
        dtypes.Date: {
            "pyarrow-nullable": "date32[pyarrow]",
        },
    }

    dtype_backend = get_dtype_backend(starting_dtype, implementation)

    for dtype_type, dtype_to_backend_mapping in dtypes_mapping.items():
        if not isinstance_or_issubclass(dtype, dtype_type):
            continue

        if dtype_backend not in dtype_to_backend_mapping:
            msg = f"Type {dtype_type} not supported for backend {dtype_backend}"
            raise NotImplementedError(msg)

        return dtype_to_backend_mapping[dtype_backend]

    msg = f"Unknown dtype: {dtype}"  # pragma: no cover
    raise AssertionError(msg)


def validate_indices(series: list[PandasSeries]) -> list[Any]:
    idx = series[0]._native_series.index
    reindexed = [series[0]._native_series]
    for s in series[1:]:
        if s._native_series.index is not idx:
            reindexed.append(
                set_axis(
                    s._native_series,
                    idx,
                    implementation=s._implementation,
                    backend_version=s._backend_version,
                )
            )
        else:
            reindexed.append(s._native_series)
    return reindexed


def to_datetime(implementation: Implementation) -> Any:
    if implementation is Implementation.PANDAS:
        return get_pandas().to_datetime
    if implementation is Implementation.MODIN:
        return get_modin().to_datetime
    if implementation is Implementation.CUDF:
        return get_cudf().to_datetime
    raise AssertionError


def int_dtype_mapper(dtype: Any) -> str:
    if "pyarrow" in str(dtype):
        return "Int64[pyarrow]"
    if str(dtype).lower() != str(dtype):  # pragma: no cover
        return "Int64"
    return "int64"


def generate_unique_token(n_bytes: int, columns: list[str]) -> str:  # pragma: no cover
    """Generates a unique token of specified n_bytes that is not present in the given list of columns.

    Arguments:
        n_bytes : The number of bytes to generate for the token.
        columns : The list of columns to check for uniqueness.

    Returns:
        A unique token that is not present in the given list of columns.

    Raises:
        AssertionError: If a unique token cannot be generated after 100 attempts.
    """
    counter = 0
    while True:
        token = secrets.token_hex(n_bytes)
        if token not in columns:
            return token

        counter += 1
        if counter > 100:
            msg = (
                "Internal Error: Narwhals was not able to generate a column name to perform cross "
                "join operation"
            )
            raise AssertionError(msg)
