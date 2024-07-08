# Utilities for expression parsing
# Useful for backends which don't have any concept of expressions, such
# and pandas or PyArrow.
from __future__ import annotations

from copy import copy
from typing import Any
from typing import Callable
from typing import Protocol
from typing import TypeAlias
from typing import TypeVar
from typing import Union

from narwhals._pandas_like.utils import create_native_series
from narwhals.dependencies import get_numpy
from narwhals.utils import flatten


class CompliantDataFrame(Protocol):
    def __narwhals_dataframe__(self) -> Any: ...


class CompliantSeries(Protocol):
    def __narwhals_series__(self) -> Any: ...


class CompliantExpr(Protocol):
    def _call(self) -> Callable[[CompliantDataFrame], list[CompliantSeries]]: ...


T = TypeVar("T")
IntoCompliantExpr: TypeAlias = Union["CompliantExpr", str, "CompliantSeries"]


def evaluate_into_expr(
    df: CompliantDataFrame, into_expr: IntoCompliantExpr
) -> list[CompliantSeries]:
    """
    Return list of raw columns.
    """
    expr = parse_into_expr(
        into_expr, implementation=df._implementation, backend_version=df._backend_version
    )
    return expr._call(df)


def evaluate_into_exprs(
    df: CompliantDataFrame,
    *exprs: IntoCompliantExpr,
    **named_exprs: IntoCompliantExpr,
) -> list[CompliantSeries]:
    """Evaluate each expr into Series."""
    series: list[CompliantSeries] = [
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


def maybe_evaluate_expr(
    df: CompliantDataFrame, expr: CompliantExpr | T
) -> list[CompliantSeries] | T:
    """Evaluate `expr` if it's an expression, otherwise return it as is."""
    if hasattr(expr, "__narwhals_expr__"):
        return expr._call(df)  # type: ignore[arg-type]
    return expr


def parse_into_exprs(
    implementation: str,
    *exprs: IntoCompliantExpr,
    backend_version: tuple[int, ...],
    **named_exprs: IntoCompliantExpr,
) -> list[CompliantExpr]:
    """Parse each input as an expression (if it's not already one). See `parse_into_expr` for
    more details."""
    out = [
        parse_into_expr(
            into_expr, implementation=implementation, backend_version=backend_version
        )
        for into_expr in flatten(exprs)
    ]
    for name, expr in named_exprs.items():
        out.append(
            parse_into_expr(
                expr, implementation=implementation, backend_version=backend_version
            ).alias(name)
        )
    return out


def parse_into_expr(
    into_expr: IntoCompliantExpr,
    *,
    implementation: str,
    backend_version: tuple[int, ...],
) -> CompliantExpr:
    """Parse `into_expr` as an expression.

    For example, in Polars, we can do both `df.select('a')` and `df.select(pl.col('a'))`.
    We do the same in Narwhals:

    - if `into_expr` is already an expression, just return it
    - if it's a Series, then convert it to an expression
    - if it's a numpy array, then convert it to a Series and then to an expression
    - if it's a string, then convert it to an expression
    - else, raise
    """
    from narwhals._arrow.expr import ArrowExpr
    from narwhals._arrow.namespace import ArrowNamespace
    from narwhals._arrow.series import ArrowSeries
    from narwhals._pandas_like.expr import PandasExpr
    from narwhals._pandas_like.namespace import PandasNamespace
    from narwhals._pandas_like.series import PandasSeries

    if implementation == "arrow":
        plx: ArrowNamespace | PandasNamespace = ArrowNamespace(
            backend_version=backend_version
        )
    else:
        plx = PandasNamespace(
            implementation=implementation, backend_version=backend_version
        )
    if isinstance(into_expr, (PandasExpr, ArrowExpr)):
        return into_expr  # type: ignore[return-value]
    if isinstance(into_expr, (PandasSeries, ArrowSeries)):
        return plx._create_expr_from_series(into_expr)  # type: ignore[arg-type, return-value]
    if isinstance(into_expr, str):
        return plx.col(into_expr)  # type: ignore[return-value]
    if (np := get_numpy()) is not None and isinstance(into_expr, np.ndarray):
        series = create_native_series(
            into_expr, implementation=implementation, backend_version=backend_version
        )
        return plx._create_expr_from_series(series)  # type: ignore[arg-type, return-value]
    msg = f"Expected IntoExpr, got {type(into_expr)}"  # pragma: no cover
    raise AssertionError(msg)


def reuse_series_implementation(
    expr: Any, attr: str, *args: Any, returns_scalar: bool = False, **kwargs: Any
) -> Any:
    """Reuse Series implementation for expression.

    If Series.foo is already defined, and we'd like Expr.foo to be the same, we can
    leverage this method to do that for us.

    Arguments
        expr: expression object.
        attr: name of method.
        returns_scalar: whether the Series version returns a scalar. In this case,
            the expression version should return a 1-row Series.
        args, kwargs: arguments and keyword arguments to pass to function.
    """
    plx = expr.__narwhals_namespace__()

    def func(df: CompliantDataFrame) -> list[CompliantSeries]:
        out: list[CompliantSeries] = []
        for column in expr._call(df):
            _out = getattr(column, attr)(
                *[maybe_evaluate_expr(df, arg) for arg in args],
                **{
                    arg_name: maybe_evaluate_expr(df, arg_value)
                    for arg_name, arg_value in kwargs.items()
                },
            )
            if returns_scalar:
                out.append(plx._create_series_from_scalar(_out, column))
            else:
                out.append(_out)
        if expr._output_names is not None:  # safety check
            assert [s.name for s in out] == expr._output_names
        return out

    # Try tracking root and output names by combining them from all
    # expressions appearing in args and kwargs. If any anonymous
    # expression appears (e.g. nw.all()), then give up on tracking root names
    # and just set it to None.
    root_names = copy(expr._root_names)
    output_names = expr._output_names
    for arg in list(args) + list(kwargs.values()):
        if root_names is not None and isinstance(arg, expr.__class__):
            if arg._root_names is not None:
                root_names.extend(arg._root_names)
            else:
                root_names = None
                output_names = None
                break
        elif root_names is None:
            output_names = None
            break

    assert (output_names is None and root_names is None) or (
        output_names is not None and root_names is not None
    )  # safety check

    return plx._create_expr_from_callable(  # type: ignore[return-value]
        func,
        depth=expr._depth + 1,
        function_name=f"{expr._function_name}->{attr}",
        root_names=root_names,
        output_names=output_names,
    )


def reuse_series_namespace_implementation(
    expr: Any, namespace: str, attr: str, *args: Any, **kwargs: Any
) -> CompliantExpr:
    """Just like `reuse_series_implementation`, but for e.g. `Expr.dt.foo` instead
    of `Expr.foo`.
    """
    from narwhals._pandas_like.expr import PandasExpr

    return PandasExpr(
        lambda df: [
            getattr(getattr(series, namespace), attr)(*args, **kwargs)
            for series in expr._call(df)
        ],
        depth=expr._depth + 1,
        function_name=f"{expr._function_name}->{namespace}.{attr}",
        root_names=expr._root_names,
        output_names=expr._output_names,
        implementation=expr._implementation,
        backend_version=expr._backend_version,
    )
