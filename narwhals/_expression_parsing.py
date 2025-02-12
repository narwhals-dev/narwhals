# Utilities for expression parsing
# Useful for backends which don't have any concept of expressions, such
# and pandas or PyArrow.
from __future__ import annotations

from enum import Enum
from enum import auto
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Sequence
from typing import TypedDict
from typing import TypeVar
from typing import Union
from typing import overload

from narwhals.dependencies import is_numpy_array
from narwhals.exceptions import LengthChangingExprError
from narwhals.exceptions import ShapeError
from narwhals.utils import Implementation
from narwhals.utils import is_compliant_expr

if TYPE_CHECKING:
    from narwhals._arrow.expr import ArrowExpr
    from narwhals._pandas_like.expr import PandasLikeExpr
    from narwhals.expr import Expr
    from narwhals.typing import CompliantDataFrame
    from narwhals.typing import CompliantExpr
    from narwhals.typing import CompliantLazyFrame
    from narwhals.typing import CompliantNamespace
    from narwhals.typing import CompliantSeries
    from narwhals.typing import CompliantSeriesT_co
    from narwhals.typing import IntoExpr

    ArrowOrPandasLikeExpr = TypeVar(
        "ArrowOrPandasLikeExpr", bound=Union[ArrowExpr, PandasLikeExpr]
    )
    PandasLikeExprT = TypeVar("PandasLikeExprT", bound=PandasLikeExpr)
    ArrowExprT = TypeVar("ArrowExprT", bound=ArrowExpr)

    T = TypeVar("T")


def evaluate_into_expr(
    df: CompliantDataFrame | CompliantLazyFrame,
    expr: CompliantExpr[CompliantSeriesT_co],
) -> Sequence[CompliantSeriesT_co]:
    """Return list of raw columns.

    This is only use for eager backends (pandas, PyArrow), where we
    alias operations at each step. As a safety precaution, here we
    can check that the expected result names match those we were
    expecting from the various `evaluate_output_names` / `alias_output_names`
    calls. Note that for PySpark / DuckDB, we are less free to liberally
    set aliases whenever we want.
    """
    _, aliases = evaluate_output_names_and_aliases(expr, df, [])
    result = expr(df)
    if list(aliases) != [s.name for s in result]:  # pragma: no cover
        msg = f"Safety assertion failed, expected {aliases}, got {result}"
        raise AssertionError(msg)
    return result


def evaluate_into_exprs(
    df: CompliantDataFrame, /, *exprs: CompliantExpr[CompliantSeriesT_co]
) -> list[CompliantSeriesT_co]:
    """Evaluate each expr into Series."""
    return [
        item
        for sublist in (evaluate_into_expr(df, into_expr) for into_expr in exprs)
        for item in sublist
    ]


@overload
def maybe_evaluate_expr(
    df: CompliantDataFrame, expr: CompliantExpr[CompliantSeriesT_co]
) -> Sequence[CompliantSeriesT_co]: ...


@overload
def maybe_evaluate_expr(df: CompliantDataFrame, expr: T) -> T: ...


def maybe_evaluate_expr(
    df: CompliantDataFrame, expr: CompliantExpr[CompliantSeriesT_co] | T
) -> Sequence[CompliantSeriesT_co] | T:
    """Evaluate `expr` if it's an expression, otherwise return it as is."""
    return expr(df) if is_compliant_expr(expr) else expr


@overload
def reuse_series_implementation(
    expr: PandasLikeExprT,
    attr: str,
    *,
    returns_scalar: bool = False,
    **kwargs: Any,
) -> PandasLikeExprT: ...


@overload
def reuse_series_implementation(
    expr: ArrowExprT,
    attr: str,
    *,
    returns_scalar: bool = False,
    **kwargs: Any,
) -> ArrowExprT: ...


def reuse_series_implementation(
    expr: ArrowExprT | PandasLikeExprT,
    attr: str,
    *,
    returns_scalar: bool = False,
    **expressifiable_args: Any,
) -> ArrowExprT | PandasLikeExprT:
    """Reuse Series implementation for expression.

    If Series.foo is already defined, and we'd like Expr.foo to be the same, we can
    leverage this method to do that for us.

    Arguments:
        expr: expression object.
        attr: name of method.
        returns_scalar: whether the Series version returns a scalar. In this case,
            the expression version should return a 1-row Series.
        args: arguments to pass to function.
        expressifiable_args: keyword arguments to pass to function, which may
            be expressifiable (e.g. `nw.col('a').is_between(3, nw.col('b')))`).
    """
    plx = expr.__narwhals_namespace__()

    def func(df: CompliantDataFrame) -> Sequence[CompliantSeries]:
        _kwargs = {
            arg_name: maybe_evaluate_expr(df, arg_value)
            for arg_name, arg_value in expressifiable_args.items()
        }

        # For PyArrow.Series, we return Python Scalars (like Polars does) instead of PyArrow Scalars.
        # However, when working with expressions, we keep everything PyArrow-native.
        extra_kwargs = (
            {"_return_py_scalar": False}
            if returns_scalar and expr._implementation is Implementation.PYARROW
            else {}
        )

        out: list[CompliantSeries] = [
            plx._create_series_from_scalar(
                getattr(series, attr)(**extra_kwargs, **_kwargs),
                reference_series=series,  # type: ignore[arg-type]
            )
            if returns_scalar
            else getattr(series, attr)(**_kwargs)
            for series in expr(df)  # type: ignore[arg-type]
        ]
        _, aliases = evaluate_output_names_and_aliases(expr, df, [])
        if [s.name for s in out] != list(aliases):  # pragma: no cover
            msg = (
                f"Safety assertion failed, please report a bug to https://github.com/narwhals-dev/narwhals/issues\n"
                f"Expression aliases: {aliases}\n"
                f"Series names: {[s.name for s in out]}"
            )
            raise AssertionError(msg)
        return out

    return plx._create_expr_from_callable(  # type: ignore[return-value]
        func,  # type: ignore[arg-type]
        depth=expr._depth + 1,
        function_name=f"{expr._function_name}->{attr}",
        evaluate_output_names=expr._evaluate_output_names,  # type: ignore[arg-type]
        alias_output_names=expr._alias_output_names,
        kwargs={**expr._kwargs, **expressifiable_args},
    )


@overload
def reuse_series_namespace_implementation(
    expr: ArrowExprT, series_namespace: str, attr: str, **kwargs: Any
) -> ArrowExprT: ...
@overload
def reuse_series_namespace_implementation(
    expr: PandasLikeExprT, series_namespace: str, attr: str, **kwargs: Any
) -> PandasLikeExprT: ...
def reuse_series_namespace_implementation(
    expr: ArrowExprT | PandasLikeExprT,
    series_namespace: str,
    attr: str,
    **kwargs: Any,
) -> ArrowExprT | PandasLikeExprT:
    """Reuse Series implementation for expression.

    Just like `reuse_series_implementation`, but for e.g. `Expr.dt.foo` instead
    of `Expr.foo`.

    Arguments:
        expr: expression object.
        series_namespace: The Series namespace (e.g. `dt`, `cat`, `str`, `list`, `name`)
        attr: name of method.
        args: arguments to pass to function.
        kwargs: keyword arguments to pass to function.
    """
    plx = expr.__narwhals_namespace__()
    return plx._create_expr_from_callable(  # type: ignore[return-value]
        lambda df: [
            getattr(getattr(series, series_namespace), attr)(**kwargs)
            for series in expr(df)  # type: ignore[arg-type]
        ],
        depth=expr._depth + 1,
        function_name=f"{expr._function_name}->{series_namespace}.{attr}",
        evaluate_output_names=expr._evaluate_output_names,  # type: ignore[arg-type]
        alias_output_names=expr._alias_output_names,
        kwargs={**expr._kwargs, **kwargs},
    )


def is_simple_aggregation(expr: CompliantExpr[Any]) -> bool:
    """Check if expr is a very simple one.

    Examples:
        - nw.col('a').mean()  # depth 1
        - nw.mean('a')  # depth 1
        - nw.len()  # depth 0

    as opposed to, say

        - nw.col('a').filter(nw.col('b')>nw.col('c')).max()

    because then, we can use a fastpath in pandas.
    """
    return expr._depth < 2


def combine_evaluate_output_names(
    *exprs: CompliantExpr[Any],
) -> Callable[[CompliantDataFrame | CompliantLazyFrame], Sequence[str]]:
    # Follow left-hand-rule for naming. E.g. `nw.sum_horizontal(expr1, expr2)` takes the
    # first name of `expr1`.
    def evaluate_output_names(
        df: CompliantDataFrame | CompliantLazyFrame,
    ) -> Sequence[str]:
        if not is_compliant_expr(exprs[0]):  # pragma: no cover
            msg = f"Safety assertion failed, expected expression, got: {type(exprs[0])}. Please report a bug."
            raise AssertionError(msg)
        return exprs[0]._evaluate_output_names(df)[:1]

    return evaluate_output_names


def combine_alias_output_names(
    *exprs: CompliantExpr[Any],
) -> Callable[[Sequence[str]], Sequence[str]] | None:
    # Follow left-hand-rule for naming. E.g. `nw.sum_horizontal(expr1.alias(alias), expr2)` takes the
    # aliasing function of `expr1` and apply it to the first output name of `expr1`.
    if exprs[0]._alias_output_names is None:
        return None

    def alias_output_names(names: Sequence[str]) -> Sequence[str]:
        return exprs[0]._alias_output_names(names)[:1]  # type: ignore[misc]

    return alias_output_names


def extract_compliant(
    plx: CompliantNamespace[CompliantSeriesT_co],
    other: Any,
    *,
    strings_are_column_names: bool,
) -> CompliantExpr[CompliantSeriesT_co] | CompliantSeriesT_co | Any:
    from narwhals.expr import Expr
    from narwhals.series import Series

    if isinstance(other, Expr):
        return other._to_compliant_expr(plx)
    if strings_are_column_names and isinstance(other, str):
        return plx.col(other)
    if isinstance(other, Series):
        return plx._create_expr_from_series(other._compliant_series)  # type: ignore[attr-defined]
    if is_numpy_array(other):
        series = plx._create_compliant_series(other)  # type: ignore[attr-defined]
        return plx._create_expr_from_series(series)  # type: ignore[attr-defined]
    return other


def evaluate_output_names_and_aliases(
    expr: CompliantExpr[Any],
    df: CompliantDataFrame | CompliantLazyFrame,
    exclude: Sequence[str],
) -> tuple[Sequence[str], Sequence[str]]:
    output_names = expr._evaluate_output_names(df)
    if not output_names:
        return [], []
    aliases = (
        output_names
        if expr._alias_output_names is None
        else expr._alias_output_names(output_names)
    )
    if expr._function_name.split("->", maxsplit=1)[0] in {"all", "selector"}:
        # For multi-output aggregations, e.g. `df.group_by('a').agg(nw.all().mean())`, we skip
        # the keys, else they would appear duplicated in the output.
        output_names, aliases = zip(
            *[(x, alias) for x, alias in zip(output_names, aliases) if x not in exclude]
        )
    return output_names, aliases


def operation_is_order_dependent(*args: IntoExpr | Any) -> bool:
    from narwhals.expr import Expr

    return any(isinstance(x, Expr) and x._metadata["is_order_dependent"] for x in args)


class ExprKind(Enum):
    """Describe which kind of expression we are dealing with.

    Commutative composition rules are:
    - LITERAL vs LITERAL -> LITERAL
    - CHANGES_LENGTH vs (LITERAL | AGGREGATION) -> CHANGES_LENGTH
    - CHANGES_LENGTH vs (CHANGES_LENGTH | TRANSFORM) -> raise
    - TRANSFORM vs (LITERAL | AGGREGATION) -> TRANSFORM
    - AGGREGATION vs (LITERAL | AGGREGATION) -> AGGREGATION
    """

    LITERAL = auto()  # e.g. nw.lit(1)
    AGGREGATION = auto()  # e.g. nw.col('a').mean()
    TRANSFORM = auto()  # length-preserving, e.g. nw.col('a').round()
    CHANGES_LENGTH = auto()  # e.g. nw.col('a').drop_nulls()


class ExprMetadata(TypedDict):
    kind: ExprKind
    is_order_dependent: bool


def combine_metadata(*args: IntoExpr, strings_are_column_names: bool) -> ExprMetadata:
    # Combine metadata from `args`.
    from narwhals.expr import Expr

    n_changes_length = 0
    has_transforms = False
    has_aggregations = False
    has_literals = False
    result_is_order_dependent = False

    for arg in args:
        if isinstance(arg, str) and strings_are_column_names:
            has_transforms = True
        elif isinstance(arg, Expr):
            if arg._metadata["is_order_dependent"]:
                result_is_order_dependent = True
            kind = arg._metadata["kind"]
            if kind is ExprKind.AGGREGATION:
                has_aggregations = True
            elif kind is ExprKind.LITERAL:
                has_literals = True
            elif kind is ExprKind.CHANGES_LENGTH:
                n_changes_length += 1
            elif kind is ExprKind.TRANSFORM:
                has_transforms = True
            else:  # pragma: no cover
                msg = "unreachable code"
                raise AssertionError(msg)
    if (
        has_literals
        and not has_aggregations
        and not has_transforms
        and not n_changes_length
    ):
        result_kind = ExprKind.LITERAL
    elif n_changes_length > 1:
        msg = "Length-changing expressions can only be used in isolation, or followed by an aggregation"
        raise LengthChangingExprError(msg)
    elif n_changes_length and (has_transforms or has_aggregations):
        msg = "Cannot combine length-changing expressions with length-preserving ones or aggregations"
        raise ShapeError(msg)
    elif n_changes_length:
        result_kind = ExprKind.CHANGES_LENGTH
    elif has_transforms:
        result_kind = ExprKind.TRANSFORM
    else:
        result_kind = ExprKind.AGGREGATION

    return ExprMetadata(kind=result_kind, is_order_dependent=result_is_order_dependent)


def check_expressions_transform(*args: IntoExpr, function_name: str) -> None:
    # Raise if any argument in `args` isn't length-preserving.
    # For Series input, we don't raise (yet), we let such checks happen later,
    # as this function works lazily and so can't evaluate lengths.
    from narwhals.expr import Expr
    from narwhals.series import Series

    if not all(
        (isinstance(x, Expr) and x._metadata["kind"] is ExprKind.TRANSFORM)
        or isinstance(x, (str, Series))
        for x in args
    ):
        msg = f"Expressions which aggregate or change length cannot be passed to '{function_name}'."
        raise ShapeError(msg)


def all_exprs_are_aggs_or_literals(*args: IntoExpr, **kwargs: IntoExpr) -> bool:
    # Raise if any argument in `args` isn't an aggregation or literal.
    # For Series input, we don't raise (yet), we let such checks happen later,
    # as this function works lazily and so can't evaluate lengths.
    from narwhals import Expr

    return all(
        isinstance(x, Expr)
        and x._metadata["kind"] in (ExprKind.AGGREGATION, ExprKind.LITERAL)
        for x in args
    ) and all(
        isinstance(x, Expr)
        and x._metadata["kind"] in (ExprKind.AGGREGATION, ExprKind.LITERAL)
        for x in kwargs.values()
    )

def infer_expr_kind(into_expr: IntoExpr) -> ExprKind:
    from narwhals.expr import Expr
    from narwhals.series import Series
    if isinstance(into_expr, Expr):
        return into_expr._metadata['kind']
    if isinstance(into_expr, (str, Series)):
        return ExprKind.TRANSFORM
    return ExprKind.LITERAL


def apply_n_ary_operation(plx: CompliantNamespace, expr: Expr, function: Callable[[Any], CompliantExpr[Any]], *comparands: IntoExpr) -> CompliantExpr[Any]:
    compliant_exprs = [
        expr._to_compliant_expr(plx),
        *(extract_compliant(plx, comparand, strings_are_column_names=False) for comparand in comparands),
    ]
    kinds = [
        expr._metadata["kind"],
        *(infer_expr_kind(comparand) for comparand in comparands)
    ]

    broadcast = any(
        kind is ExprKind.TRANSFORM for kind in kinds
    )
    compliant_exprs = [
        compliant_expr.broadcast(kind)
        # `compliant` expr could also be literal, hence the check is needed before calling `broadcast`.
        # We can't (yet) use `plx.lit` for all literals due to dtype mismatches in pandas,
        # see test failures in https://github.com/narwhals-dev/narwhals/pull/1999.
        if broadcast and is_compliant_expr(compliant_expr) and compliant_expr in (ExprKind.AGGREGATION, ExprKind.LITERAL)
        else compliant_expr
        for compliant_expr, kind in zip(compliant_exprs, kinds)
    ]
    return function(*compliant_exprs)
