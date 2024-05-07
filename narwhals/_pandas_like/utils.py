from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable
from typing import TypeVar

from narwhals.dependencies import get_cudf
from narwhals.dependencies import get_modin
from narwhals.dependencies import get_pandas
from narwhals.dependencies import get_pyarrow
from narwhals.utils import flatten
from narwhals.utils import isinstance_or_issubclass
from narwhals.utils import parse_version

T = TypeVar("T")

if TYPE_CHECKING:
    from narwhals._pandas_like.dataframe import PandasDataFrame
    from narwhals._pandas_like.expr import PandasExpr
    from narwhals._pandas_like.series import PandasSeries
    from narwhals.dtypes import DType

    ExprT = TypeVar("ExprT", bound=PandasExpr)

    from narwhals._pandas_like.typing import IntoPandasExpr


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
        if other._series.index is not index and not (other._series.index == index).all():
            return other._series.set_axis(index, axis=0)
        return other._series
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
            return item(other._series)
        if other._series.index is not index and not (other._series.index == index).all():
            return other._series.set_axis(index, axis=0)
        return other._series
    raise AssertionError("Please report a bug")


def maybe_evaluate_expr(df: PandasDataFrame, arg: Any) -> Any:
    """Evaluate expression if it's an expression, otherwise return it as is."""
    from narwhals._pandas_like.expr import PandasExpr

    if isinstance(arg, PandasExpr):
        return arg._call(df)
    return arg


def parse_into_exprs(
    implementation: str,
    *exprs: IntoPandasExpr | Iterable[IntoPandasExpr],
    **named_exprs: IntoPandasExpr,
) -> list[PandasExpr]:
    out = [parse_into_expr(implementation, into_expr) for into_expr in flatten(exprs)]
    for name, expr in named_exprs.items():
        out.append(parse_into_expr(implementation, expr).alias(name))
    return out


def parse_into_expr(implementation: str, into_expr: IntoPandasExpr) -> PandasExpr:
    from narwhals._pandas_like.expr import PandasExpr
    from narwhals._pandas_like.namespace import PandasNamespace
    from narwhals._pandas_like.series import PandasSeries

    plx = PandasNamespace(implementation=implementation)

    if isinstance(into_expr, PandasExpr):
        return into_expr
    if isinstance(into_expr, PandasSeries):
        return plx._create_expr_from_series(into_expr)
    if isinstance(into_expr, str):
        return plx.col(into_expr)
    msg = f"Expected IntoExpr, got {type(into_expr)}"  # pragma: no cover
    raise AssertionError(msg)


def evaluate_into_expr(
    df: PandasDataFrame, into_expr: IntoPandasExpr
) -> list[PandasSeries]:
    """
    Return list of raw columns.
    """
    expr = parse_into_expr(df._implementation, into_expr)
    return expr._call(df)


def evaluate_into_exprs(
    df: PandasDataFrame,
    *exprs: IntoPandasExpr | Iterable[IntoPandasExpr],
    **named_exprs: IntoPandasExpr,
) -> list[PandasSeries]:
    """Evaluate each expr into Series."""
    series: list[PandasSeries] = [
        item
        for sublist in [evaluate_into_expr(df, into_expr) for into_expr in flatten(exprs)]
        for item in sublist
    ]
    for name, expr in named_exprs.items():
        evaluated_expr = evaluate_into_expr(df, expr)
        if len(evaluated_expr) > 1:
            msg = "Named expressions must return a single column"  # pragma: no cover
            raise AssertionError(msg)
        series.append(evaluated_expr[0].alias(name))
    return series


def register_expression_call(expr: ExprT, attr: str, *args: Any, **kwargs: Any) -> ExprT:
    from narwhals._pandas_like.expr import PandasExpr
    from narwhals._pandas_like.namespace import PandasNamespace
    from narwhals._pandas_like.series import PandasSeries

    plx = PandasNamespace(implementation=expr._implementation)

    def func(df: PandasDataFrame) -> list[PandasSeries]:
        out: list[PandasSeries] = []
        for column in expr._call(df):
            _out = getattr(column, attr)(
                *[maybe_evaluate_expr(df, arg) for arg in args],
                **{
                    arg_name: maybe_evaluate_expr(df, arg_value)
                    for arg_name, arg_value in kwargs.items()
                },
            )
            if isinstance(_out, PandasSeries):
                out.append(_out)
            else:
                out.append(plx._create_series_from_scalar(_out, column))
        if expr._output_names is not None:
            assert [s._series.name for s in out] == expr._output_names
        return out

    root_names = copy(expr._root_names)
    for arg in list(args) + list(kwargs.values()):
        if root_names is not None and isinstance(arg, PandasExpr):
            if arg._root_names is not None:
                root_names.extend(arg._root_names)
            else:
                root_names = None
                break
        elif root_names is None:
            break

    return plx._create_expr_from_callable(  # type: ignore[return-value]
        func,
        depth=expr._depth + 1,
        function_name=f"{expr._function_name}->{attr}",
        root_names=root_names,
        output_names=expr._output_names,
    )


def item(s: Any) -> Any:
    # cuDF doesn't have Series.item().
    if len(s) != 1:
        msg = "Can only convert a Series of length 1 to a scalar"  # pragma: no cover
        raise AssertionError(msg)
    return s.iloc[0]


def is_simple_aggregation(expr: PandasExpr) -> bool:
    return (
        expr._function_name is not None
        and expr._depth is not None
        and expr._depth < 2
        # todo: avoid this one?
        and (expr._root_names is not None or (expr._depth == 0))
    )


def horizontal_concat(dfs: list[Any], implementation: str) -> Any:
    """
    Concatenate (native) DataFrames horizontally.

    Should be in namespace.
    """
    if implementation == "pandas":
        import pandas as pd

        if parse_version(pd.__version__) < parse_version("3.0.0"):
            return pd.concat(dfs, axis=1, copy=False)
        return pd.concat(dfs, axis=1)  # pragma: no cover
    if implementation == "cudf":  # pragma: no cover
        import cudf

        return cudf.concat(dfs, axis=1)
    if implementation == "modin":  # pragma: no cover
        import modin.pandas as mpd

        return mpd.concat(dfs, axis=1)
    msg = f"Unknown implementation: {implementation}"  # pragma: no cover
    raise TypeError(msg)  # pragma: no cover


def vertical_concat(dfs: list[Any], implementation: str) -> Any:
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
    if implementation == "pandas":
        import pandas as pd

        if parse_version(pd.__version__) < parse_version("3.0.0"):
            return pd.concat(dfs, axis=0, copy=False)
        return pd.concat(dfs, axis=0)  # pragma: no cover
    if implementation == "cudf":  # pragma: no cover
        import cudf

        return cudf.concat(dfs, axis=0)
    if implementation == "modin":  # pragma: no cover
        import modin.pandas as mpd

        return mpd.concat(dfs, axis=0)
    msg = f"Unknown implementation: {implementation}"  # pragma: no cover
    raise TypeError(msg)  # pragma: no cover


def series_from_iterable(
    data: Iterable[Any], name: str, index: Any, implementation: str
) -> Any:
    """Return native series."""
    if implementation == "pandas":
        import pandas as pd

        return pd.Series(data, name=name, index=index, copy=False)
    if implementation == "cudf":  # pragma: no cover
        import cudf

        return cudf.Series(data, name=name, index=index)
    if implementation == "modin":  # pragma: no cover
        import modin.pandas as mpd

        return mpd.Series(data, name=name, index=index)
    msg = f"Unknown implementation: {implementation}"  # pragma: no cover
    raise TypeError(msg)  # pragma: no cover


def translate_dtype(dtype: Any) -> DType:
    from narwhals import dtypes

    if dtype in ("int64", "Int64", "Int64[pyarrow]"):
        return dtypes.Int64()
    if dtype in ("int32", "Int32", "Int32[pyarrow]"):
        return dtypes.Int32()
    if dtype in ("int16", "Int16", "Int16[pyarrow]"):
        return dtypes.Int16()
    if dtype in ("int8", "Int8", "Int8[pyarrow]"):
        return dtypes.Int8()
    if dtype in ("uint64", "UInt64", "UInt64[pyarrow]"):
        return dtypes.UInt64()
    if dtype in ("uint32", "UInt32", "UInt32[pyarrow]"):
        return dtypes.UInt32()
    if dtype in ("uint16", "UInt16", "UInt16[pyarrow]"):
        return dtypes.UInt16()
    if dtype in ("uint8", "UInt8", "UInt8[pyarrow]"):
        return dtypes.UInt8()
    if dtype in ("float64", "Float64", "Float64[pyarrow]"):
        return dtypes.Float64()
    if dtype in ("float32", "Float32", "Float32[pyarrow]"):
        return dtypes.Float32()
    if dtype in ("string", "string[python]", "string[pyarrow]"):
        return dtypes.String()
    if dtype in ("bool", "boolean", "boolean[pyarrow]"):
        return dtypes.Boolean()
    if str(dtype).startswith("datetime64"):
        # todo: different time units and time zones
        return dtypes.Datetime()
    if dtype == "object":
        return dtypes.String()
    msg = f"Unknown dtype: {dtype}"  # pragma: no cover
    raise AssertionError(msg)


def reverse_translate_dtype(dtype: DType | type[DType]) -> Any:
    # Use the default pandas dtype here
    # TODO: maybe this could be configurable?
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
        import pandas as pd

        if parse_version(pd.__version__) >= parse_version("2.0.0"):
            if get_pyarrow() is not None:
                return "string[pyarrow]"
            return "string[python]"  # pragma: no cover
        return "object"  # pragma: no cover
    if isinstance_or_issubclass(dtype, dtypes.Boolean):
        return "bool"
    if isinstance_or_issubclass(dtype, dtypes.Datetime):
        # todo: different time units and time zones
        return "datetime64[us]"
    msg = f"Unknown dtype: {dtype}"  # pragma: no cover
    raise AssertionError(msg)


def validate_indices(series: list[PandasSeries]) -> list[Any]:
    idx = series[0]._series.index
    reindexed = [series[0]._series]
    for s in series[1:]:
        if s._series.index is not idx and not (s._series.index == idx).all():
            reindexed.append(s._series.set_axis(idx.rename(s._series.index.name), axis=0))
        else:
            reindexed.append(s._series)
    return reindexed


def to_datetime(implementation: str) -> Any:
    if implementation == "pandas":
        return get_pandas().to_datetime
    if implementation == "modin":
        return get_modin().to_datetime
    if implementation == "cudf":
        return get_cudf().to_datetime
    raise AssertionError
