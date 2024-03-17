from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable
from typing import TypeVar

from narwhals.utils import flatten
from narwhals.utils import remove_prefix

T = TypeVar("T")

if TYPE_CHECKING:
    from narwhals.dtypes import DType
    from narwhals.pandas_like.dataframe import PandasDataFrame
    from narwhals.pandas_like.expr import PandasExpr
    from narwhals.pandas_like.series import PandasSeries

    ExprT = TypeVar("ExprT", bound=PandasExpr)

    from narwhals.pandas_like.typing import IntoPandasExpr


def validate_column_comparand(other: Any) -> Any:
    """Validate RHS of binary operation.

    If the comparison isn't supported, return `NotImplemented` so that the
    "right-hand-side" operation (e.g. `__radd__`) can be tried.

    If RHS is length 1, return the scalar value, so that the underlying
    library can broadcast it.
    """
    from narwhals.pandas_like.dataframe import PandasDataFrame
    from narwhals.pandas_like.series import PandasSeries

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
        return other._series
    return other


def validate_dataframe_comparand(other: Any) -> Any:
    """Validate RHS of binary operation.

    If the comparison isn't supported, return `NotImplemented` so that the
    "right-hand-side" operation (e.g. `__radd__`) can be tried.
    """
    from narwhals.pandas_like.dataframe import PandasDataFrame
    from narwhals.pandas_like.series import PandasSeries

    if isinstance(other, list) and len(other) > 1:
        # e.g. `plx.all() + plx.all()`
        msg = "Multi-output expressions are not supported in this context"
        raise ValueError(msg)
    if isinstance(other, list):
        other = other[0]
    if isinstance(other, PandasDataFrame):
        return NotImplemented
    if isinstance(other, PandasSeries):
        if other.len() == 1:
            # broadcast
            return item(other._series)
        return other._series
    return other


def maybe_evaluate_expr(df: PandasDataFrame, arg: Any) -> Any:
    """Evaluate expression if it's an expression, otherwise return it as is."""
    from narwhals.pandas_like.expr import PandasExpr

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
    from narwhals.pandas_like.expr import PandasExpr
    from narwhals.pandas_like.namespace import PandasNamespace
    from narwhals.pandas_like.series import PandasSeries

    plx = PandasNamespace(implementation=implementation)

    if isinstance(into_expr, PandasExpr):
        return into_expr
    if isinstance(into_expr, PandasSeries):
        return plx._create_expr_from_series(into_expr)
    if isinstance(into_expr, str):
        return plx.col(into_expr)
    msg = f"Expected IntoExpr, got {type(into_expr)}"
    raise TypeError(msg)


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
            msg = "Named expressions must return a single column"
            raise ValueError(msg)
        series.append(evaluated_expr[0].alias(name))
    return series


def register_expression_call(expr: ExprT, attr: str, *args: Any, **kwargs: Any) -> ExprT:
    from narwhals.pandas_like.expr import PandasExpr
    from narwhals.pandas_like.namespace import PandasNamespace
    from narwhals.pandas_like.series import PandasSeries

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
        msg = "Can only convert a Series of length 1 to a scalar"
        raise ValueError(msg)
    return s.iloc[0]


def is_simple_aggregation(expr: PandasExpr) -> bool:
    return (
        expr._function_name is not None
        and expr._depth is not None
        and expr._depth < 2
        # todo: avoid this one?
        and expr._root_names is not None
    )


def evaluate_simple_aggregation(expr: PandasExpr, grouped: Any) -> Any:
    """
    Use fastpath for simple aggregations if possible.

    If an aggregation is simple (e.g. `pl.col('a').mean()`), then pandas-like
    implementations have a fastpath we can use.

    For example, `df.group_by('a').agg(pl.col('b').mean())` can be evaluated
    as `df.groupby('a')['b'].mean()`, whereas
    `df.group_by('a').agg(mean=(pl.col('b') - pl.col('c').mean()).mean())`
    requires a lambda function, which is slower.

    Returns naive DataFrame.
    """
    if expr._root_names is None or expr._output_names is None:
        msg = "Expected expr to have root_names and output_names set, but they are None. Please report a bug."
        raise AssertionError(msg)
    if len(expr._root_names) != len(expr._output_names):
        msg = "Expected expr to have same number of root_names and output_names, but they are different. Please report a bug."
        raise AssertionError(msg)
    new_names = dict(zip(expr._root_names, expr._output_names))
    function_name = remove_prefix(expr._function_name, "col->")
    return getattr(grouped[expr._root_names], function_name)()[expr._root_names].rename(
        columns=new_names
    )


def horizontal_concat(dfs: list[Any], implementation: str) -> Any:
    """
    Concatenate (native) DataFrames.

    Should be in namespace.
    """
    if implementation == "pandas":
        import pandas as pd

        return pd.concat(dfs, axis=1, copy=False)
    if implementation == "cudf":
        import cudf

        return cudf.concat(dfs, axis=1)
    if implementation == "modin":
        import modin.pandas as mpd

        return mpd.concat(dfs, axis=1)
    msg = f"Unknown implementation: {implementation}"
    raise TypeError(msg)


def dataframe_from_dict(data: dict[str, Any], implementation: str) -> Any:
    """Return native dataframe."""
    if implementation == "pandas":
        import pandas as pd

        return pd.DataFrame(data, copy=False)
    if implementation == "cudf":
        import cudf

        return cudf.DataFrame(data)
    if implementation == "modin":
        import modin.pandas as mpd

        return mpd.DataFrame(data)
    msg = f"Unknown implementation: {implementation}"
    raise TypeError(msg)


def series_from_iterable(
    data: Iterable[Any], name: str, index: Any, implementation: str
) -> Any:
    """Return native series."""
    if implementation == "pandas":
        import pandas as pd

        return pd.Series(data, name=name, index=index, copy=False)
    if implementation == "cudf":
        import cudf

        return cudf.Series(data, name=name, index=index)
    if implementation == "modin":
        import modin.pandas as mpd

        return mpd.Series(data, name=name, index=index)
    msg = f"Unknown implementation: {implementation}"
    raise TypeError(msg)


def translate_dtype(dtype: Any) -> DType:
    from narwhals import dtypes

    if dtype in ("int64", "Int64"):
        return dtypes.Int64()
    if dtype == "int32":
        return dtypes.Int32()
    if dtype == "int16":
        return dtypes.Int16()
    if dtype == "int8":
        return dtypes.Int8()
    if dtype == "uint64":
        return dtypes.Int64()
    if dtype == "uint32":
        return dtypes.UInt32()
    if dtype == "uint16":
        return dtypes.UInt16()
    if dtype == "uint8":
        return dtypes.UInt8()
    if dtype in ("float64", "Float64"):
        return dtypes.Float64()
    if dtype in ("float32", "Float32"):
        return dtypes.Float32()
    if dtype == ("string"):
        return dtypes.String()
    if dtype in ("bool", "boolean"):
        return dtypes.Boolean()
    if dtype == "object":
        return dtypes.Object()
    if str(dtype).startswith("datetime64"):
        return dtypes.Datetime()
    msg = f"Unknown dtype: {dtype}"
    raise TypeError(msg)


def isinstance_or_issubclass(obj: Any, cls: Any) -> bool:
    return isinstance(obj, cls) or issubclass(obj, cls)


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
    if isinstance_or_issubclass(dtype, dtypes.UInt8):
        return "uint8"
    if isinstance_or_issubclass(dtype, dtypes.UInt64):
        return "uint64"
    if isinstance_or_issubclass(dtype, dtypes.UInt32):
        return "uint32"
    if isinstance_or_issubclass(dtype, dtypes.UInt16):
        return "uint16"
    if isinstance_or_issubclass(dtype, dtypes.UInt8):
        return "uint8"
    if isinstance_or_issubclass(dtype, dtypes.String):
        return "object"
    if isinstance_or_issubclass(dtype, dtypes.Boolean):
        return "bool"
    msg = f"Unknown dtype: {dtype}"
    raise TypeError(msg)


def reset_index(obj: Any) -> Any:
    index = obj.index
    if (
        hasattr(index, "start")
        and hasattr(index, "stop")
        and hasattr(index, "step")
        and index.start == 0
        and index.stop == len(obj)
        and index.step == 1
    ):
        return obj
    return obj.reset_index(drop=True)
