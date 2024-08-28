from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from narwhals.dependencies import get_pandas
from narwhals.dependencies import get_pyarrow
from narwhals.utils import isinstance_or_issubclass
from narwhals.utils import parse_version

if TYPE_CHECKING:
    import dask.dataframe as dd
    import dask_expr

    from narwhals._dask.dataframe import DaskLazyFrame
    from narwhals.dtypes import DType


def maybe_evaluate(df: DaskLazyFrame, obj: Any) -> Any:
    from narwhals._dask.expr import DaskExpr

    if isinstance(obj, DaskExpr):
        results = obj._call(df)
        if len(results) != 1:  # pragma: no cover
            msg = "Multi-output expressions not supported in this context"
            raise NotImplementedError(msg)
        result = results[0]
        validate_comparand(df._native_frame, result)
        if obj._returns_scalar:
            # Return scalar, let Dask do its broadcasting
            return result[0]
        return result
    return obj


def parse_exprs_and_named_exprs(
    df: DaskLazyFrame, *exprs: Any, **named_exprs: Any
) -> dict[str, dask_expr.Series]:
    results = {}
    for expr in exprs:
        if hasattr(expr, "__narwhals_expr__"):
            _results = expr._call(df)
        elif isinstance(expr, str):
            _results = [df._native_frame.loc[:, expr]]
        else:  # pragma: no cover
            msg = f"Expected expression or column name, got: {expr}"
            raise TypeError(msg)
        for _result in _results:
            if getattr(expr, "_returns_scalar", False):
                results[_result.name] = _result[0]
            else:
                results[_result.name] = _result
    for name, value in named_exprs.items():
        _results = value._call(df)
        if len(_results) != 1:  # pragma: no cover
            msg = "Named expressions must return a single column"
            raise AssertionError(msg)
        for _result in _results:
            if getattr(value, "_returns_scalar", False):
                results[name] = _result[0]
            else:
                results[name] = _result
    return results


def add_row_index(frame: dd.DataFrame, name: str) -> dd.DataFrame:
    frame = frame.assign(**{name: 1})
    return frame.assign(**{name: frame[name].cumsum(method="blelloch") - 1})


def validate_comparand(lhs: dask_expr.Series, rhs: dask_expr.Series) -> None:
    import dask_expr  # ignore-banned-import

    if not dask_expr._expr.are_co_aligned(lhs._expr, rhs._expr):  # pragma: no cover
        # are_co_aligned is a method which cheaply checks if two Dask expressions
        # have the same index, and therefore don't require index alignment.
        # If someone only operates on a Dask DataFrame via expressions, then this
        # should always be the case: expression outputs (by definition) all come from the
        # same input dataframe, and Dask Series does not have any operations which
        # change the index. Nonetheless, we perform this safety check anyway.

        # However, we still need to carefully vet which methods we support for Dask, to
        # avoid issues where `are_co_aligned` doesn't do what we want it to do:
        # https://github.com/dask/dask-expr/issues/1112.
        msg = "Objects are not co-aligned, so this operation is not supported for Dask backend"
        raise RuntimeError(msg)


def reverse_translate_dtype(dtype: DType | type[DType]) -> Any:
    from narwhals import dtypes

    if isinstance_or_issubclass(dtype, dtypes.Float64):
        return "float64"
    if isinstance_or_issubclass(dtype, dtypes.Float32):
        return "float32"
    if isinstance_or_issubclass(dtype, dtypes.Int64):
        return "int64"
    if isinstance_or_issubclass(dtype, dtypes.Int32):
        return "int32"
    if isinstance_or_issubclass(dtype, dtypes.Int16):
        return "int16"
    if isinstance_or_issubclass(dtype, dtypes.Int8):
        return "int8"
    if isinstance_or_issubclass(dtype, dtypes.UInt64):
        return "uint64"
    if isinstance_or_issubclass(dtype, dtypes.UInt32):
        return "uint32"
    if isinstance_or_issubclass(dtype, dtypes.UInt16):
        return "uint16"
    if isinstance_or_issubclass(dtype, dtypes.UInt8):
        return "uint8"
    if isinstance_or_issubclass(dtype, dtypes.String):
        if (pd := get_pandas()) is not None and parse_version(
            pd.__version__
        ) >= parse_version("2.0.0"):
            if get_pyarrow() is not None:
                return "string[pyarrow]"
            return "string[python]"  # pragma: no cover
        return "object"  # pragma: no cover
    if isinstance_or_issubclass(dtype, dtypes.Boolean):
        return "bool"
    if isinstance_or_issubclass(dtype, dtypes.Categorical):
        return "category"
    if isinstance_or_issubclass(dtype, dtypes.Datetime):
        return "datetime64[us]"
    if isinstance_or_issubclass(dtype, dtypes.Duration):
        return "timedelta64[ns]"

    msg = f"Unknown dtype: {dtype}"  # pragma: no cover
    raise AssertionError(msg)
