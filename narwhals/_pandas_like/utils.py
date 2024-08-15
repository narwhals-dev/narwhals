from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable
from typing import TypeVar

from narwhals.dependencies import get_cudf
from narwhals.dependencies import get_modin
from narwhals.dependencies import get_pandas
from narwhals.utils import Implementation
from narwhals.utils import isinstance_or_issubclass

T = TypeVar("T")

if TYPE_CHECKING:
    from narwhals._pandas_like.expr import PandasLikeExpr
    from narwhals._pandas_like.series import PandasLikeSeries
    from narwhals.dtypes import DType

    ExprT = TypeVar("ExprT", bound=PandasLikeExpr)


def validate_column_comparand(index: Any, other: Any) -> Any:
    """Validate RHS of binary operation.

    If the comparison isn't supported, return `NotImplemented` so that the
    "right-hand-side" operation (e.g. `__radd__`) can be tried.

    If RHS is length 1, return the scalar value, so that the underlying
    library can broadcast it.
    """
    from narwhals._pandas_like.dataframe import PandasLikeDataFrame
    from narwhals._pandas_like.series import PandasLikeSeries

    if isinstance(other, list):
        if len(other) > 1:
            # e.g. `plx.all() + plx.all()`
            msg = "Multi-output expressions are not supported in this context"
            raise ValueError(msg)
        other = other[0]
    if isinstance(other, PandasLikeDataFrame):
        return NotImplemented
    if isinstance(other, PandasLikeSeries):
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
    from narwhals._pandas_like.dataframe import PandasLikeDataFrame
    from narwhals._pandas_like.series import PandasLikeSeries

    if isinstance(other, PandasLikeDataFrame):
        return NotImplemented
    if isinstance(other, PandasLikeSeries):
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
    msg = "Please report a bug"  # pragma: no cover
    raise AssertionError(msg)


def create_native_series(
    iterable: Any,
    index: Any = None,
    *,
    implementation: Implementation,
    backend_version: tuple[int, ...],
) -> PandasLikeSeries:
    from narwhals._pandas_like.series import PandasLikeSeries

    if implementation is Implementation.PANDAS:
        pd = get_pandas()
        series = pd.Series(iterable, index=index, name="")
    elif implementation is Implementation.MODIN:
        mpd = get_modin()
        series = mpd.Series(iterable, index=index, name="")
    elif implementation is Implementation.CUDF:
        cudf = get_cudf()
        series = cudf.Series(iterable, index=index, name="")
    return PandasLikeSeries(
        series, implementation=implementation, backend_version=backend_version
    )


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


def translate_dtype(column: Any) -> DType:
    from narwhals import dtypes

    dtype = column.dtype
    if str(dtype) in ("int64", "Int64", "Int64[pyarrow]", "int64[pyarrow]"):
        return dtypes.Int64()
    if str(dtype) in ("int32", "Int32", "Int32[pyarrow]", "int32[pyarrow]"):
        return dtypes.Int32()
    if str(dtype) in ("int16", "Int16", "Int16[pyarrow]", "int16[pyarrow]"):
        return dtypes.Int16()
    if str(dtype) in ("int8", "Int8", "Int8[pyarrow]", "int8[pyarrow]"):
        return dtypes.Int8()
    if str(dtype) in ("uint64", "UInt64", "UInt64[pyarrow]", "uint64[pyarrow]"):
        return dtypes.UInt64()
    if str(dtype) in ("uint32", "UInt32", "UInt32[pyarrow]", "uint32[pyarrow]"):
        return dtypes.UInt32()
    if str(dtype) in ("uint16", "UInt16", "UInt16[pyarrow]", "uint16[pyarrow]"):
        return dtypes.UInt16()
    if str(dtype) in ("uint8", "UInt8", "UInt8[pyarrow]", "uint8[pyarrow]"):
        return dtypes.UInt8()
    if str(dtype) in (
        "float64",
        "Float64",
        "Float64[pyarrow]",
        "float64[pyarrow]",
        "double[pyarrow]",
    ):
        return dtypes.Float64()
    if str(dtype) in (
        "float32",
        "Float32",
        "Float32[pyarrow]",
        "float32[pyarrow]",
        "float[pyarrow]",
    ):
        return dtypes.Float32()
    if str(dtype) in (
        "string",
        "string[python]",
        "string[pyarrow]",
        "large_string[pyarrow]",
    ):
        return dtypes.String()
    if str(dtype) in ("bool", "boolean", "boolean[pyarrow]", "bool[pyarrow]"):
        return dtypes.Boolean()
    if str(dtype) in ("category",) or str(dtype).startswith("dictionary<"):
        return dtypes.Categorical()
    if str(dtype).startswith("datetime64"):
        # TODO(Unassigned): different time units and time zones
        return dtypes.Datetime()
    if str(dtype).startswith("timedelta64") or str(dtype).startswith("duration"):
        # TODO(Unassigned): different time units
        return dtypes.Duration()
    if str(dtype).startswith("timestamp["):
        # pyarrow-backed datetime
        # TODO(Unassigned): different time units and time zones
        return dtypes.Datetime()
    if str(dtype) == "date32[day][pyarrow]":
        return dtypes.Date()
    if str(dtype) == "object":
        if (  # pragma: no cover  TODO(unassigned): why does this show as uncovered?
            idx := getattr(column, "first_valid_index", lambda: None)()
        ) is not None and isinstance(column.loc[idx], str):
            # Infer based on first non-missing value.
            # For pandas pre 3.0, this isn't perfect.
            # After pandas 3.0, pandas has a dedicated string dtype
            # which is inferred by default.
            return dtypes.String()
        else:
            df = column.to_frame()
            if hasattr(df, "__dataframe__"):
                from narwhals._interchange.dataframe import (
                    map_interchange_dtype_to_narwhals_dtype,
                )

                try:
                    return map_interchange_dtype_to_narwhals_dtype(
                        df.__dataframe__().get_column(0).dtype
                    )
                except Exception:  # noqa: BLE001
                    return dtypes.Object()
            else:  # pragma: no cover
                return dtypes.Object()
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


def narwhals_to_native_dtype(  # noqa: PLR0915
    dtype: DType | type[DType], starting_dtype: Any, implementation: Implementation
) -> Any:
    from narwhals import dtypes

    if "polars" in str(type(dtype)):
        msg = (
            f"Expected Narwhals object, got: {type(dtype)}.\n\n"
            "Perhaps you:\n"
            "- Forgot a `nw.from_native` somewhere?\n"
            "- Used `pl.Int64` instead of `nw.Int64`?"
        )
        raise TypeError(msg)

    dtype_backend = get_dtype_backend(starting_dtype, implementation)
    if isinstance_or_issubclass(dtype, dtypes.Float64):
        if dtype_backend == "pyarrow-nullable":
            return "Float64[pyarrow]"
        if dtype_backend == "pandas-nullable":
            return "Float64"
        else:
            return "float64"
    if isinstance_or_issubclass(dtype, dtypes.Float32):
        if dtype_backend == "pyarrow-nullable":
            return "Float32[pyarrow]"
        if dtype_backend == "pandas-nullable":
            return "Float32"
        else:
            return "float32"
    if isinstance_or_issubclass(dtype, dtypes.Int64):
        if dtype_backend == "pyarrow-nullable":
            return "Int64[pyarrow]"
        if dtype_backend == "pandas-nullable":
            return "Int64"
        else:
            return "int64"
    if isinstance_or_issubclass(dtype, dtypes.Int32):
        if dtype_backend == "pyarrow-nullable":
            return "Int32[pyarrow]"
        if dtype_backend == "pandas-nullable":
            return "Int32"
        else:
            return "int32"
    if isinstance_or_issubclass(dtype, dtypes.Int16):
        if dtype_backend == "pyarrow-nullable":
            return "Int16[pyarrow]"
        if dtype_backend == "pandas-nullable":
            return "Int16"
        else:
            return "int16"
    if isinstance_or_issubclass(dtype, dtypes.Int8):
        if dtype_backend == "pyarrow-nullable":
            return "Int8[pyarrow]"
        if dtype_backend == "pandas-nullable":
            return "Int8"
        else:
            return "int8"
    if isinstance_or_issubclass(dtype, dtypes.UInt64):
        if dtype_backend == "pyarrow-nullable":
            return "UInt64[pyarrow]"
        if dtype_backend == "pandas-nullable":
            return "UInt64"
        else:
            return "uint64"
    if isinstance_or_issubclass(dtype, dtypes.UInt32):
        if dtype_backend == "pyarrow-nullable":
            return "UInt32[pyarrow]"
        if dtype_backend == "pandas-nullable":
            return "UInt32"
        else:
            return "uint32"
    if isinstance_or_issubclass(dtype, dtypes.UInt16):
        if dtype_backend == "pyarrow-nullable":
            return "UInt16[pyarrow]"
        if dtype_backend == "pandas-nullable":
            return "UInt16"
        else:
            return "uint16"
    if isinstance_or_issubclass(dtype, dtypes.UInt8):
        if dtype_backend == "pyarrow-nullable":
            return "UInt8[pyarrow]"
        if dtype_backend == "pandas-nullable":
            return "UInt8"
        else:
            return "uint8"
    if isinstance_or_issubclass(dtype, dtypes.String):
        if dtype_backend == "pyarrow-nullable":
            return "string[pyarrow]"
        if dtype_backend == "pandas-nullable":
            return "string"
        else:
            return str
    if isinstance_or_issubclass(dtype, dtypes.Boolean):
        if dtype_backend == "pyarrow-nullable":
            return "boolean[pyarrow]"
        if dtype_backend == "pandas-nullable":
            return "boolean"
        else:
            return "bool"
    if isinstance_or_issubclass(dtype, dtypes.Categorical):
        # TODO(Unassigned): is there no pyarrow-backed categorical?
        # or at least, convert_dtypes(dtype_backend='pyarrow') doesn't
        # convert to it?
        return "category"
    if isinstance_or_issubclass(dtype, dtypes.Datetime):
        # TODO(Unassigned): different time units and time zones
        if dtype_backend == "pyarrow-nullable":
            return "timestamp[ns][pyarrow]"
        return "datetime64[ns]"
    if isinstance_or_issubclass(dtype, dtypes.Duration):
        # TODO(Unassigned): different time units and time zones
        if dtype_backend == "pyarrow-nullable":
            return "duration[ns][pyarrow]"
        return "timedelta64[ns]"
    if isinstance_or_issubclass(dtype, dtypes.Date):
        if dtype_backend == "pyarrow-nullable":
            return "date32[pyarrow]"
        msg = "Date dtype only supported for pyarrow-backed data types in pandas"
        raise NotImplementedError(msg)
    if isinstance_or_issubclass(dtype, dtypes.Enum):
        msg = "Converting to Enum is not (yet) supported"
        raise NotImplementedError(msg)
    msg = f"Unknown dtype: {dtype}"  # pragma: no cover
    raise AssertionError(msg)


def broadcast_series(series: list[PandasLikeSeries]) -> list[Any]:
    native_namespace = series[0].__native_namespace__()

    lengths = [len(s) for s in series]
    max_length = max(lengths)

    idx = series[lengths.index(max_length)]._native_series.index
    reindexed = []

    for s, length in zip(series, lengths):
        s_native = s._native_series
        if max_length > 1 and length == 1:
            reindexed.append(
                native_namespace.Series(
                    [s_native.iloc[0]] * max_length,
                    index=idx,
                    name=s_native.name,
                    dtype=s_native.dtype,
                )
            )

        elif s_native.index is not idx:
            reindexed.append(
                set_axis(
                    s_native,
                    idx,
                    implementation=s._implementation,
                    backend_version=s._backend_version,
                )
            )
        else:
            reindexed.append(s_native)
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
