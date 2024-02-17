from __future__ import annotations
from typing import Any, Iterable, cast, TypeVar

from polars_api_compat.spec import (
    DataFrame,
    LazyFrame,
    Series,
    Expr,
    IntoExpr,
    Namespace,
)

ExprT = TypeVar("ExprT", bound=Expr)

T = TypeVar("T")


def validate_column_comparand(column: Any, other: Any) -> Any:
    """Validate RHS of binary operation.

    If the comparison isn't supported, return `NotImplemented` so that the
    "right-hand-side" operation (e.g. `__radd__`) can be tried.

    If RHS is length 1, return the scalar value, so that the underlying
    library can broadcast it.
    """
    if isinstance(other, list):
        if len(other) > 1:
            # e.g. `plx.all() + plx.all()`
            raise ValueError(
                "Multi-output expressions are not supported in this context"
            )
        other = other[0]
    if hasattr(
        other,
        "__dataframe_namespace__",
    ):
        return NotImplemented
    if hasattr(other, "__series_namespace__"):
        if other.len() == 1:
            # broadcast
            return other.item()
        if (
            hasattr(column.series, "index")
            and hasattr(other.series, "index")
            and column.series.index is not other.series.index
        ):
            msg = (
                "Left index is not right index. "
                "You were probably trying to compare different dataframes "
                "without first having joined them. Either join them, or "
                "consider using expressions."
            )
            raise ValueError(msg)
        return other.series
    return other


def validate_dataframe_comparand(dataframe: Any, other: Any) -> Any:
    """Validate RHS of binary operation.

    If the comparison isn't supported, return `NotImplemented` so that the
    "right-hand-side" operation (e.g. `__radd__`) can be tried.
    """
    if isinstance(other, list) and len(other) > 1:
        # e.g. `plx.all() + plx.all()`
        raise ValueError("Multi-output expressions are not supported in this context")
    elif isinstance(other, list) and len(other) == 1:
        other = other[0]
    if hasattr(
        other,
        "__dataframe_namespace__",
    ):
        return NotImplemented
    if hasattr(other, "__series_namespace__"):
        if other.len() == 1:
            # broadcast
            return other.get_value(0)
        if (
            hasattr(dataframe.dataframe, "index")
            and hasattr(other.series, "index")
            and dataframe.dataframe.index is not other.series.index
        ):
            msg = (
                "Left index is not right index. "
                "You were probably trying to compare different dataframes "
                "without first having joined them. Either join them, or "
                "consider using expressions."
            )
            raise ValueError(msg)
        return other.series
    return other


def maybe_evaluate_expr(df: DataFrame | LazyFrame, arg: Any) -> Any:
    """Evaluate expression if it's an expression, otherwise return it as is."""
    if hasattr(arg, "__expr_namespace__"):
        return arg.call(df)
    return arg


def get_namespace(df: DataFrame | LazyFrame) -> Namespace:
    if hasattr(df, "__dataframe_namespace__"):
        return df.__dataframe_namespace__()
    if hasattr(df, "__lazyframe_namespace__"):
        return df.__lazyframe_namespace__()
    raise TypeError(f"Expected DataFrame or LazyFrame, got {type(df)}")


def parse_into_exprs(
    plx: Namespace, *exprs: IntoExpr | Iterable[IntoExpr], **named_exprs: IntoExpr
) -> list[Expr]:
    out = [parse_into_expr(plx, into_expr) for into_expr in flatten_into_expr(*exprs)]
    for name, expr in named_exprs.items():
        out.append(parse_into_expr(plx, expr).alias(name))
    return out


def parse_into_expr(plx: Namespace, into_expr: IntoExpr) -> Expr:
    if isinstance(into_expr, str):
        return plx.col(into_expr)
    if hasattr(into_expr, "__expr_namespace__"):
        return cast(Expr, into_expr)  # help mypy
    if hasattr(into_expr, "__series_namespace__"):
        into_expr = cast(Series, into_expr)  # help mypy
        return plx._create_expr_from_series(into_expr)
    raise TypeError(f"Expected IntoExpr, got {type(into_expr)}")


def evaluate_into_expr(df: DataFrame | LazyFrame, into_expr: IntoExpr) -> list[Series]:
    """
    Return list of raw columns.
    """
    expr = parse_into_expr(get_namespace(df), into_expr)
    return expr.call(df)


def flatten_str(*args: str | Iterable[str]) -> list[str]:
    out: list[str] = []
    for arg in args:
        if isinstance(arg, str):
            out.append(arg)
        else:
            out.extend(arg)
    return out


def flatten_bool(*args: bool | Iterable[bool]) -> list[bool]:
    out: list[bool] = []
    for arg in args:
        if isinstance(arg, bool):
            out.append(arg)
        else:
            out.extend(arg)
    return out


def flatten_into_expr(*args: IntoExpr | Iterable[IntoExpr]) -> list[IntoExpr]:
    out: list[IntoExpr] = []
    for arg in args:
        if isinstance(arg, (list, tuple)):
            out.extend(arg)
        else:
            out.append(arg)  # type: ignore[arg-type]
    return out


# in filter, I want to:
# - flatten the into exprs
# - convert all to exprs
# - pass these to all_horizontal


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
            raise ValueError("Named expressions must return a single column")
        series.append(evaluated_expr[0].alias(name))
    return series


def register_expression_call(expr: Expr, attr: str, *args: Any, **kwargs: Any) -> Expr:
    plx = expr.__expr_namespace__()

    def func(df: DataFrame | LazyFrame) -> list[Series]:
        out: list[Series] = []
        for column in expr.call(df):
            # should be enough to just evaluate?
            # validation should happen within column methods?
            _out = getattr(column, attr)(  # type: ignore[no-any-return]
                *[maybe_evaluate_expr(df, arg) for arg in args],
                **{
                    arg_name: maybe_evaluate_expr(df, arg_value)
                    for arg_name, arg_value in kwargs.items()
                },
            )
            if hasattr(_out, "__series_namespace__"):
                _out = cast(Series, _out)  # help mypy
                out.append(_out)
            else:
                out.append(plx._create_series_from_scalar(_out, column))
        return out

    if expr.depth is None:
        raise AssertionError("Unreachable code, please report a bug")
    if expr.function_name is not None:
        function_name: str | None = f"{expr.function_name}->{attr}"
    else:
        function_name = expr.function_name
    return plx._create_expr_from_callable(
        func,
        depth=expr.depth + 1,
        function_name=function_name,
        root_names=expr.root_names,
        output_names=expr.output_names,
    )
