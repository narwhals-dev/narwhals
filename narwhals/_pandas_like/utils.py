from __future__ import annotations

import re
from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable
from typing import Literal
from typing import TypeVar

from narwhals._arrow.utils import (
    native_to_narwhals_dtype as arrow_native_to_narwhals_dtype,
)
from narwhals.dependencies import get_polars
from narwhals.utils import Implementation
from narwhals.utils import isinstance_or_issubclass

T = TypeVar("T")

if TYPE_CHECKING:
    from narwhals._pandas_like.expr import PandasLikeExpr
    from narwhals._pandas_like.series import PandasLikeSeries
    from narwhals.dtypes import DType
    from narwhals.typing import DTypes

    ExprT = TypeVar("ExprT", bound=PandasLikeExpr)
    import pandas as pd


PANDAS_LIKE_IMPLEMENTATION = {
    Implementation.PANDAS,
    Implementation.CUDF,
    Implementation.MODIN,
}
PD_DATETIME_RGX = r"""^
    datetime64\[
        (?P<time_unit>s|ms|us|ns)                 # Match time unit: s, ms, us, or ns
        (?:,                                      # Begin non-capturing group for optional timezone
            \s*                                   # Optional whitespace after comma
            (?P<time_zone>                        # Start named group for timezone
                [a-zA-Z\/]+                       # Match timezone name, e.g., UTC, America/New_York
                (?:[+-]\d{2}:\d{2})?              # Optional offset in format +HH:MM or -HH:MM
                |                                 # OR
                pytz\.FixedOffset\(\d+\)          # Match pytz.FixedOffset with integer offset in parentheses
            )                                     # End time_zone group
        )?                                        # End optional timezone group
    \]                                            # Closing bracket for datetime64
$"""
PATTERN_PD_DATETIME = re.compile(PD_DATETIME_RGX, re.VERBOSE)
PA_DATETIME_RGX = r"""^
    timestamp\[
        (?P<time_unit>s|ms|us|ns)                 # Match time unit: s, ms, us, or ns
        (?:,                                      # Begin non-capturing group for optional timezone
            \s?tz=                                # Match "tz=" prefix
            (?P<time_zone>                        # Start named group for timezone
                [a-zA-Z\/]*                       # Match timezone name (e.g., UTC, America/New_York)
                (?:                               # Begin optional non-capturing group for offset
                    [+-]\d{2}:\d{2}               # Match offset in format +HH:MM or -HH:MM
                )?                                # End optional offset group
            )                                     # End time_zone group
        )?                                        # End optional timezone group
    \]                                            # Closing bracket for timestamp
    \[pyarrow\]                                   # Literal string "[pyarrow]"
$"""
PATTERN_PA_DATETIME = re.compile(PA_DATETIME_RGX, re.VERBOSE)
PD_DURATION_RGX = r"""^
    timedelta64\[
        (?P<time_unit>s|ms|us|ns)                 # Match time unit: s, ms, us, or ns
    \]                                            # Closing bracket for timedelta64
$"""

PATTERN_PD_DURATION = re.compile(PD_DURATION_RGX, re.VERBOSE)
PA_DURATION_RGX = r"""^
    duration\[
        (?P<time_unit>s|ms|us|ns)                 # Match time unit: s, ms, us, or ns
    \]                                            # Closing bracket for duration
    \[pyarrow\]                                   # Literal string "[pyarrow]"
$"""
PATTERN_PA_DURATION = re.compile(PA_DURATION_RGX, re.VERBOSE)


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
            s = other._native_series
            return s.__class__(s.iloc[0], index=index, dtype=s.dtype)
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
            s = other._native_series
            return s.__class__(s.iloc[0], index=index, dtype=s.dtype, name=s.name)
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
    dtypes: DTypes,
) -> PandasLikeSeries:
    from narwhals._pandas_like.series import PandasLikeSeries

    if implementation in PANDAS_LIKE_IMPLEMENTATION:
        series = implementation.to_native_namespace().Series(
            iterable, index=index, name=""
        )
        return PandasLikeSeries(
            series,
            implementation=implementation,
            backend_version=backend_version,
            dtypes=dtypes,
        )
    else:  # pragma: no cover
        msg = f"Expected pandas-like implementation ({PANDAS_LIKE_IMPLEMENTATION}), found {implementation}"
        raise TypeError(msg)


def horizontal_concat(
    dfs: list[Any], *, implementation: Implementation, backend_version: tuple[int, ...]
) -> Any:
    """
    Concatenate (native) DataFrames horizontally.

    Should be in namespace.
    """
    if implementation in PANDAS_LIKE_IMPLEMENTATION:
        extra_kwargs = (
            {"copy": False}
            if implementation is Implementation.PANDAS and backend_version < (3,)
            else {}
        )
        return implementation.to_native_namespace().concat(dfs, axis=1, **extra_kwargs)

    else:  # pragma: no cover
        msg = f"Expected pandas-like implementation ({PANDAS_LIKE_IMPLEMENTATION}), found {implementation}"
        raise TypeError(msg)


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

    if implementation in PANDAS_LIKE_IMPLEMENTATION:
        extra_kwargs = (
            {"copy": False}
            if implementation is Implementation.PANDAS and backend_version < (3,)
            else {}
        )
        return implementation.to_native_namespace().concat(dfs, axis=0, **extra_kwargs)

    else:  # pragma: no cover
        msg = f"Expected pandas-like implementation ({PANDAS_LIKE_IMPLEMENTATION}), found {implementation}"
        raise TypeError(msg)


def native_series_from_iterable(
    data: Iterable[Any],
    name: str,
    index: Any,
    implementation: Implementation,
) -> Any:
    """Return native series."""
    if implementation in PANDAS_LIKE_IMPLEMENTATION:
        extra_kwargs = {"copy": False} if implementation is Implementation.PANDAS else {}
        return implementation.to_native_namespace().Series(
            data, name=name, index=index, **extra_kwargs
        )

    else:  # pragma: no cover
        msg = f"Expected pandas-like implementation ({PANDAS_LIKE_IMPLEMENTATION}), found {implementation}"
        raise TypeError(msg)


def set_axis(
    obj: T,
    index: Any,
    *,
    implementation: Implementation,
    backend_version: tuple[int, ...],
) -> T:
    if implementation is Implementation.CUDF:  # pragma: no cover
        obj = obj.copy(deep=False)  # type: ignore[attr-defined]
        obj.index = index  # type: ignore[attr-defined]
        return obj
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
    return obj.set_axis(index, axis=0, **kwargs)  # type: ignore[attr-defined, no-any-return]


def native_to_narwhals_dtype(
    native_column: Any, dtypes: DTypes, implementation: Implementation
) -> DType:
    dtype = str(native_column.dtype)

    if dtype in {"int64", "Int64", "Int64[pyarrow]", "int64[pyarrow]"}:
        return dtypes.Int64()
    if dtype in {"int32", "Int32", "Int32[pyarrow]", "int32[pyarrow]"}:
        return dtypes.Int32()
    if dtype in {"int16", "Int16", "Int16[pyarrow]", "int16[pyarrow]"}:
        return dtypes.Int16()
    if dtype in {"int8", "Int8", "Int8[pyarrow]", "int8[pyarrow]"}:
        return dtypes.Int8()
    if dtype in {"uint64", "UInt64", "UInt64[pyarrow]", "uint64[pyarrow]"}:
        return dtypes.UInt64()
    if dtype in {"uint32", "UInt32", "UInt32[pyarrow]", "uint32[pyarrow]"}:
        return dtypes.UInt32()
    if dtype in {"uint16", "UInt16", "UInt16[pyarrow]", "uint16[pyarrow]"}:
        return dtypes.UInt16()
    if dtype in {"uint8", "UInt8", "UInt8[pyarrow]", "uint8[pyarrow]"}:
        return dtypes.UInt8()
    if dtype in {
        "float64",
        "Float64",
        "Float64[pyarrow]",
        "float64[pyarrow]",
        "double[pyarrow]",
    }:
        return dtypes.Float64()
    if dtype in {
        "float32",
        "Float32",
        "Float32[pyarrow]",
        "float32[pyarrow]",
        "float[pyarrow]",
    }:
        return dtypes.Float32()
    if dtype in {"string", "string[python]", "string[pyarrow]", "large_string[pyarrow]"}:
        return dtypes.String()
    if dtype in {"bool", "boolean", "boolean[pyarrow]", "bool[pyarrow]"}:
        return dtypes.Boolean()
    if dtype == "category" or dtype.startswith("dictionary<"):
        return dtypes.Categorical()
    if (match_ := PATTERN_PD_DATETIME.match(dtype)) or (
        match_ := PATTERN_PA_DATETIME.match(dtype)
    ):
        dt_time_unit: Literal["us", "ns", "ms", "s"] = match_.group("time_unit")  # type: ignore[assignment]
        dt_time_zone: str | None = match_.group("time_zone")
        return dtypes.Datetime(dt_time_unit, dt_time_zone)
    if (match_ := PATTERN_PD_DURATION.match(dtype)) or (
        match_ := PATTERN_PA_DURATION.match(dtype)
    ):
        du_time_unit: Literal["us", "ns", "ms", "s"] = match_.group("time_unit")  # type: ignore[assignment]
        return dtypes.Duration(du_time_unit)
    if dtype == "date32[day][pyarrow]":
        return dtypes.Date()
    if dtype.startswith(("large_list", "list", "struct", "fixed_size_list")):
        return arrow_native_to_narwhals_dtype(native_column.dtype.pyarrow_dtype, dtypes)
    if dtype == "object":
        if implementation is Implementation.DASK:
            # Dask columns are lazy, so we can't inspect values.
            # The most useful assumption is probably String
            return dtypes.String()
        if implementation is Implementation.PANDAS:  # pragma: no cover
            # This is the most efficient implementation for pandas,
            # and doesn't require the interchange protocol
            import pandas as pd  # ignore-banned-import

            dtype = pd.api.types.infer_dtype(native_column, skipna=True)
            if dtype == "string":
                return dtypes.String()
            return dtypes.Object()
        else:  # pragma: no cover
            df = native_column.to_frame()
            if hasattr(df, "__dataframe__"):
                from narwhals._interchange.dataframe import (
                    map_interchange_dtype_to_narwhals_dtype,
                )

                try:
                    return map_interchange_dtype_to_narwhals_dtype(
                        df.__dataframe__().get_column(0).dtype, dtypes
                    )
                except Exception:  # noqa: BLE001, S110
                    pass
    return dtypes.Unknown()


def get_dtype_backend(dtype: Any, implementation: Implementation) -> str:
    if implementation is Implementation.PANDAS:
        import pandas as pd  # ignore-banned-import()

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
    dtype: DType | type[DType],
    starting_dtype: Any,
    implementation: Implementation,
    backend_version: tuple[int, ...],
    dtypes: DTypes,
) -> Any:
    if (pl := get_polars()) is not None and isinstance(
        dtype, (pl.DataType, pl.DataType.__class__)
    ):
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
        dt_time_unit = getattr(dtype, "time_unit", "us")
        dt_time_zone = getattr(dtype, "time_zone", None)

        # Pandas does not support "ms" or "us" time units before version 2.0
        # Let's overwrite with "ns"
        if implementation is Implementation.PANDAS and backend_version < (
            2,
        ):  # pragma: no cover
            dt_time_unit = "ns"

        if dtype_backend == "pyarrow-nullable":
            tz_part = f", tz={dt_time_zone}" if dt_time_zone else ""
            return f"timestamp[{dt_time_unit}{tz_part}][pyarrow]"
        else:
            tz_part = f", {dt_time_zone}" if dt_time_zone else ""
            return f"datetime64[{dt_time_unit}{tz_part}]"
    if isinstance_or_issubclass(dtype, dtypes.Duration):
        du_time_unit = getattr(dtype, "time_unit", "us")
        if implementation is Implementation.PANDAS and backend_version < (
            2,
        ):  # pragma: no cover
            dt_time_unit = "ns"
        return (
            f"duration[{du_time_unit}][pyarrow]"
            if dtype_backend == "pyarrow-nullable"
            else f"timedelta64[{du_time_unit}]"
        )

    if isinstance_or_issubclass(dtype, dtypes.Date):
        if dtype_backend == "pyarrow-nullable":
            return "date32[pyarrow]"
        msg = "Date dtype only supported for pyarrow-backed data types in pandas"
        raise NotImplementedError(msg)
    if isinstance_or_issubclass(dtype, dtypes.Enum):
        msg = "Converting to Enum is not (yet) supported"
        raise NotImplementedError(msg)
    if isinstance_or_issubclass(dtype, dtypes.List):  # pragma: no cover
        msg = "Converting to List dtype is not supported yet"
        return NotImplementedError(msg)
    if isinstance_or_issubclass(dtype, dtypes.Struct):  # pragma: no cover
        msg = "Converting to Struct dtype is not supported yet"
        return NotImplementedError(msg)
    if isinstance_or_issubclass(dtype, dtypes.Array):  # pragma: no cover
        msg = "Converting to Array dtype is not supported yet"
        return NotImplementedError(msg)
    msg = f"Unknown dtype: {dtype}"  # pragma: no cover
    raise AssertionError(msg)


def broadcast_series(series: list[PandasLikeSeries]) -> list[Any]:
    native_namespace = series[0].__native_namespace__()

    lengths = [len(s) for s in series]
    max_length = max(lengths)

    idx = series[lengths.index(max_length)]._native_series.index
    reindexed = []
    max_length_gt_1 = max_length > 1
    for s, length in zip(series, lengths):
        s_native = s._native_series
        if max_length_gt_1 and length == 1:
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
    if implementation in PANDAS_LIKE_IMPLEMENTATION:
        return implementation.to_native_namespace().to_datetime

    else:  # pragma: no cover
        msg = f"Expected pandas-like implementation ({PANDAS_LIKE_IMPLEMENTATION}), found {implementation}"
        raise TypeError(msg)


def int_dtype_mapper(dtype: Any) -> str:
    if "pyarrow" in str(dtype):
        return "Int64[pyarrow]"
    if str(dtype).lower() != str(dtype):  # pragma: no cover
        return "Int64"
    return "int64"


def convert_str_slice_to_int_slice(
    str_slice: slice, columns: pd.Index
) -> tuple[int | None, int | None, int | None]:
    start = columns.get_loc(str_slice.start) if str_slice.start is not None else None
    stop = columns.get_loc(str_slice.stop) + 1 if str_slice.stop is not None else None
    step = str_slice.step
    return (start, stop, step)


def calculate_timestamp_datetime(
    s: pd.Series, original_time_unit: str, time_unit: str
) -> pd.Series:
    if original_time_unit == "ns":
        if time_unit == "ns":
            result = s
        elif time_unit == "us":
            result = s // 1_000
        else:
            result = s // 1_000_000
    elif original_time_unit == "us":
        if time_unit == "ns":
            result = s * 1_000
        elif time_unit == "us":
            result = s
        else:
            result = s // 1_000
    elif original_time_unit == "ms":
        if time_unit == "ns":
            result = s * 1_000_000
        elif time_unit == "us":
            result = s * 1_000
        else:
            result = s
    elif original_time_unit == "s":
        if time_unit == "ns":
            result = s * 1_000_000_000
        elif time_unit == "us":
            result = s * 1_000_000
        else:
            result = s * 1_000
    else:  # pragma: no cover
        msg = f"unexpected time unit {original_time_unit}, please report a bug at https://github.com/narwhals-dev/narwhals"
        raise AssertionError(msg)
    return result


def calculate_timestamp_date(s: pd.Series, time_unit: str) -> pd.Series:
    s = s * 86_400  # number of seconds in a day
    if time_unit == "ns":
        result = s * 1_000_000_000
    elif time_unit == "us":
        result = s * 1_000_000
    else:
        result = s * 1_000
    return result
