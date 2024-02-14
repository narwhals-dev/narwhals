from __future__ import annotations
from typing import Any


# Technically, it would be possible to correctly type hint this function
# with a tonne of overloads, but for now, it' not worth it, just use Any
def validate_comparand(left: Any, right: Any) -> Any:
    """Validate comparand, raising if it can't be compared with `left`.

    If the comparison isn't supported, return `NotImplemented` so that the
    "right-hand-side" operation (e.g. `__radd__`) can be tried.
    """
    if hasattr(left, "__dataframe_namespace__") and hasattr(
        right,
        "__column_expr_namespace__",
    ):
        called = right.call(left)
        if len(called) > 1:
            raise ValueError("Multi-output expressions are not supported in this context")
        return validate_comparand(left, called[0])
    if hasattr(left, "__dataframe_namespace__") and hasattr(
        right,
        "__scalar_expr_namespace__",
    ):
        called = right.call(left)
        if len(called) > 1:
            raise ValueError("Multi-output expressions are not supported in this context")
        return validate_comparand(left, called[0])
    if hasattr(left, "__dataframe_namespace__") and hasattr(
        right,
        "__dataframe_namespace__",
    ):  # pragma: no cover
        # Technically, currently unreachable - but, keeping this in case it
        # becomes reachable in the future.
        msg = "Cannot compare different dataframe objects - please join them first"
        raise ValueError(msg)
    if hasattr(left, "__dataframe_namespace__") and hasattr(
        right,
        "__column_namespace__",
    ):
        if right.len().scalar == 1:
            # broadcast
            return right.get_value(0).scalar
        if (
            hasattr(left.dataframe, "index")
            and hasattr(right.column, "index")
            and left.dataframe.index is not right.column.index
        ):
            msg = (
                "Left index is not right index. "
                "You were probably trying to compare different dataframes "
                "without first having joined them. Either join them, or "
                "consider using expressions."
            )
            raise ValueError(msg)
        return right.column
    if hasattr(left, "__dataframe_namespace__") and hasattr(
        right,
        "__scalar_namespace__",
    ):
        return right.scalar

    if hasattr(left, "__column_namespace__") and hasattr(
        right,
        "__dataframe_namespace__",
    ):
        return NotImplemented
    if hasattr(left, "__column_namespace__") and hasattr(right, "__column_namespace__"):
        if (
            hasattr(left.column, "index")
            and hasattr(right.column, "index")
            and left.column.index is not right.column.index
        ):
            msg = (
                "Left index is not right index. "
                "You were probably trying to compare different dataframes "
                "without first having joined them. Either join them, or "
                "consider using expressions."
            )
            raise ValueError(msg)
        return right.column
    if hasattr(left, "__column_namespace__") and hasattr(right, "__scalar_namespace__"):
        return right.scalar
    if hasattr(left, "__scalar_namespace__") and hasattr(
        right,
        "__dataframe_namespace__",
    ):
        return NotImplemented
    if hasattr(left, "__scalar_namespace__") and hasattr(right, "__column_namespace__"):
        return NotImplemented
    if hasattr(left, "__scalar_namespace__") and hasattr(right, "__scalar_namespace__"):
        return right.scalar

    return right

def parse_expr(df, expr):
    """
    Return list of raw columns.
    """
    pdx = df.__dataframe_namespace__()
    if isinstance(expr, str):
        return pdx.col(expr).call(df)
    if hasattr(expr, '__column_expr_namespace__'):
        return expr.call(df)
    if isinstance(expr, (list, tuple)):
        out = []
        for _expr in expr:
            out.extend(parse_expr(df, _expr))
        return out
    raise TypeError(f"Expected str, ColumnExpr, or list/tuple of str/ColumnExpr, got {type(expr)}")



def parse_exprs(df, *exprs, **named_exprs) -> dict[str, Any]:
    """
    Take exprs and evaluate Series underneath them.
    
    Returns dict of output name to raw column object.
    """
    exprs = [parse_expr(df, expr) for expr in exprs]
    exprs = [item for sublist in exprs for item in sublist]
    named_exprs = {name: parse_expr(df, expr)[0] for name, expr in named_exprs.items()}
    new_cols = {}
    for expr in exprs:
        _series = validate_comparand(df, expr)
        new_cols[_series.name] = _series
    for name, expr in named_exprs.items():
        _series = validate_comparand(df, expr)
        new_cols[name] = _series
    return new_cols


def register_expression_call(expr: ColumnExpr, attr: str, *args, **kwargs) -> ColumnExpr:  # type: ignore[override]
    plx = expr.__column_expr_namespace__()
    def func(df: DataFrame) -> list[Column]:
        out = []
        for column in expr.call(df):
            _out = getattr(column, attr)(  # type: ignore[no-any-return]
                *[validate_comparand(df, arg) for arg in args],
                **{
                    arg_name: validate_comparand(df, arg_value)
                    for arg_name, arg_value in kwargs.items()
                },
            )
            if not hasattr(_out, "__column_namespace__"):
                out.append(plx.create_column_from_scalar(_out, column))
            else:
                out.append(_out)
        return out
    return plx.create_column_expr(func)
