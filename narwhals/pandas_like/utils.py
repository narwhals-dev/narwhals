from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable
from typing import TypeVar

T = TypeVar("T")

if TYPE_CHECKING:
    from narwhals.pandas_like.dataframe import DataFrame
    from narwhals.pandas_like.dataframe import LazyFrame
    from narwhals.pandas_like.expr import Expr
    from narwhals.pandas_like.namespace import Namespace
    from narwhals.pandas_like.series import Series

    ExprT = TypeVar("ExprT", bound=Expr)

    from narwhals.spec import IntoExpr


def validate_column_comparand(other: Any) -> Any:
    """Validate RHS of binary operation.

    If the comparison isn't supported, return `NotImplemented` so that the
    "right-hand-side" operation (e.g. `__radd__`) can be tried.

    If RHS is length 1, return the scalar value, so that the underlying
    library can broadcast it.
    """
    from narwhals.pandas_like.dataframe import DataFrame
    from narwhals.pandas_like.series import Series

    if isinstance(other, list):
        if len(other) > 1:
            # e.g. `plx.all() + plx.all()`
            msg = "Multi-output expressions are not supported in this context"
            raise ValueError(msg)
        other = other[0]
    if isinstance(other, DataFrame):
        return NotImplemented
    if isinstance(other, Series):
        if other.len() == 1:
            # broadcast
            return other.item()
        return other.series
    return other


def validate_dataframe_comparand(other: Any) -> Any:
    """Validate RHS of binary operation.

    If the comparison isn't supported, return `NotImplemented` so that the
    "right-hand-side" operation (e.g. `__radd__`) can be tried.
    """
    from narwhals.pandas_like.dataframe import DataFrame
    from narwhals.pandas_like.series import Series

    if isinstance(other, list) and len(other) > 1:
        # e.g. `plx.all() + plx.all()`
        msg = "Multi-output expressions are not supported in this context"
        raise ValueError(msg)
    if isinstance(other, list):
        other = other[0]
    if isinstance(other, DataFrame):
        return NotImplemented
    if isinstance(other, Series):
        if other.len() == 1:
            # broadcast
            return item(other)
        return other.series
    return other


def maybe_evaluate_expr(df: DataFrame | LazyFrame, arg: Any) -> Any:
    """Evaluate expression if it's an expression, otherwise return it as is."""
    from narwhals.pandas_like.expr import Expr

    if isinstance(arg, Expr):
        return arg._call(df)
    return arg


def get_namespace(obj: Any) -> Namespace:
    from narwhals.pandas_like.namespace import Namespace

    return Namespace(api_version="0.20.0", implementation=obj._implementation)


def parse_into_exprs(
    plx: Namespace, *exprs: IntoExpr | Iterable[IntoExpr], **named_exprs: IntoExpr
) -> list[Expr]:
    out = [parse_into_expr(plx, into_expr) for into_expr in flatten_into_expr(*exprs)]
    for name, expr in named_exprs.items():
        out.append(parse_into_expr(plx, expr).alias(name))
    return out


def parse_into_expr(plx: Namespace, into_expr: IntoExpr) -> Expr:
    from narwhals.pandas_like.expr import Expr
    from narwhals.pandas_like.series import Series

    if isinstance(into_expr, str):
        return plx.col(into_expr)
    if isinstance(into_expr, Expr):
        return into_expr
    if isinstance(into_expr, Series):
        return plx._create_expr_from_series(into_expr)
    msg = f"Expected IntoExpr, got {type(into_expr)}"
    raise TypeError(msg)


def evaluate_into_expr(df: DataFrame | LazyFrame, into_expr: IntoExpr) -> list[Series]:
    """
    Return list of raw columns.
    """
    expr = parse_into_expr(get_namespace(df), into_expr)
    return expr._call(df)


def flatten_str(*args: str | Iterable[str]) -> list[str]:
    out: list[str] = []
    for arg in args:
        if isinstance(arg, str):
            out.append(arg)
        else:
            for item in arg:
                if not isinstance(item, str):
                    msg = f"Expected str, got {type(item)}"
                    raise TypeError(msg)
                out.append(item)
    return out


def flatten_bool(*args: bool | Iterable[bool]) -> list[bool]:
    out: list[bool] = []
    for arg in args:
        if isinstance(arg, bool):
            out.append(arg)
        else:
            for item in arg:
                if not isinstance(item, bool):
                    msg = f"Expected str, got {type(item)}"
                    raise TypeError(msg)
                out.append(item)
    return out


def flatten_into_expr(*args: IntoExpr | Iterable[IntoExpr]) -> list[IntoExpr]:
    out: list[IntoExpr] = []
    for arg in args:
        if isinstance(arg, (list, tuple)):
            out.extend(arg)
        else:
            out.append(arg)  # type: ignore[arg-type]
    return out


def evaluate_into_exprs(
    df: DataFrame | LazyFrame,
    *exprs: IntoExpr | Iterable[IntoExpr],
    **named_exprs: IntoExpr,
) -> list[Series]:
    """Evaluate each expr into Series."""
    series: list[Series] = [
        item
        for sublist in [
            evaluate_into_expr(df, into_expr) for into_expr in flatten_into_expr(*exprs)
        ]
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
    from narwhals.pandas_like.series import Series

    plx = get_namespace(expr)

    def func(df: DataFrame | LazyFrame) -> list[Series]:
        out: list[Series] = []
        for column in expr._call(df):
            _out = getattr(column, attr)(
                *[maybe_evaluate_expr(df, arg) for arg in args],
                **{
                    arg_name: maybe_evaluate_expr(df, arg_value)
                    for arg_name, arg_value in kwargs.items()
                },
            )
            if isinstance(_out, Series):
                out.append(_out)
            else:
                out.append(plx._create_series_from_scalar(_out, column))
        return out

    if expr._function_name is not None:
        function_name: str = f"{expr._function_name}->{attr}"
    else:
        function_name = attr
    return plx._create_expr_from_callable(  # type: ignore[return-value]
        func,
        depth=expr._depth + 1,
        function_name=function_name,
        root_names=expr._root_names,
        output_names=expr._output_names,
    )


def item(s: Any) -> Any:
    # cuDF doesn't have Series.item().
    if len(s) != 1:
        msg = "Can only convert a Series of length 1 to a scalar"
        raise ValueError(msg)
    return s.iloc[0]


def is_simple_aggregation(expr: Expr) -> bool:
    return (
        expr._function_name is not None
        and expr._depth is not None
        and expr._depth < 2
        # todo: avoid this one?
        and expr._root_names is not None
    )


def evaluate_simple_aggregation(expr: Expr, grouped: Any) -> Any:
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
    assert expr._function_name is not None
    return getattr(grouped[expr._root_names], expr._function_name)()[
        expr._root_names
    ].rename(columns=new_names)


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
