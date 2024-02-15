from __future__ import annotations
from typing import Any

# TODO: split this up!
# Have a validate_column_comparand and validate_dataframe_comparand
# for the column one:
# - if it's a 1-row column, return the value
# - if it's a multi-row column, return the raw column if it's the same
#   length
# - if it's a scalar, return it
def validate_column_comparand(column: Any, other: Any) -> Any:
    """Validate RHS of binary operation.

    If the comparison isn't supported, return `NotImplemented` so that the
    "right-hand-side" operation (e.g. `__radd__`) can be tried.
    """
    if isinstance(other, list) and len(other) > 1:
        raise ValueError("Multi-output expressions are not supported in this context")
    elif isinstance(other, list) and len(other) == 1:
        other = other[0]
    if hasattr(
        other,
        "__dataframe_namespace__",
    ):
        return NotImplemented
    if hasattr(other, "__column_namespace__"):
        if other.len() == 1:
            # broadcast
            return other.get_value(0)
        if (
            hasattr(column.column, "index")
            and hasattr(other.column, "index")
            and column.column.index is not other.column.index
        ):
            msg = (
                "Left index is not right index. "
                "You were probably trying to compare different dataframes "
                "without first having joined them. Either join them, or "
                "consider using expressions."
            )
            raise ValueError(msg)
        return other.column
    return other

def evaluate_expr(df, expr):
    if hasattr(expr, '__column_expr_namespace__'):
        return expr.call(df)
    return expr

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
    parsed_exprs = [item for sublist in [parse_expr(df, expr) for expr in exprs] for item in sublist]
    parsed_named_exprs = {}
    for name, expr in named_exprs.items():
        parsed_expr = parse_expr(df, expr)
        if len(parsed_expr) > 1:
            raise ValueError("Named expressions must return a single column")
        parsed_named_exprs[name] = parsed_expr[0]
    new_cols = {}
    for expr in parsed_exprs:
        _column = evaluate_expr(df, expr)
        new_cols[_column.name] = _column
    for name, expr in parsed_named_exprs.items():
        _column = evaluate_expr(df, expr)
        new_cols[name] = _column
    return new_cols


def register_expression_call(expr: ColumnExpr, attr: str, *args, **kwargs) -> ColumnExpr:  # type: ignore[override]
    plx = expr.__column_expr_namespace__()
    def func(df: DataFrame) -> list[Column]:
        out = []
        for column in expr.call(df):
            # should be enough to just evaluate?
            # validation should happen within column methods?
            _out = getattr(column, attr)(  # type: ignore[no-any-return]
                *[evaluate_expr(df, arg) for arg in args],
                **{
                    arg_name: evaluate_expr(df, arg_value)
                    for arg_name, arg_value in kwargs.items()
                },
            )
            if hasattr(_out, "__column_namespace__"):
                out.append(_out)
            else:
                out.append(plx.create_column_from_scalar(_out, column))
        return out
    return plx.create_column_expr(func)
