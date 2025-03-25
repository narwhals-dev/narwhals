# Utilities for expression parsing
# Useful for backends which don't have any concept of expressions, such
# and pandas or PyArrow.
from __future__ import annotations

from enum import Enum
from enum import auto
from itertools import chain
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Literal
from typing import Sequence
from typing import TypeVar
from typing import cast

from narwhals.dependencies import is_narwhals_series
from narwhals.dependencies import is_numpy_array
from narwhals.exceptions import LengthChangingExprError
from narwhals.exceptions import MultiOutputExpressionError
from narwhals.exceptions import ShapeError
from narwhals.utils import is_compliant_expr

if TYPE_CHECKING:
    from typing_extensions import Never
    from typing_extensions import TypeIs

    from narwhals._compliant import CompliantExpr
    from narwhals._compliant import CompliantExprT
    from narwhals._compliant import CompliantFrameT
    from narwhals._compliant import CompliantNamespace
    from narwhals._compliant.typing import EagerNamespaceAny
    from narwhals.expr import Expr
    from narwhals.typing import CompliantDataFrame
    from narwhals.typing import CompliantLazyFrame
    from narwhals.typing import IntoExpr
    from narwhals.typing import _1DArray

    T = TypeVar("T")


def is_expr(obj: Any) -> TypeIs[Expr]:
    """Check whether `obj` is a Narwhals Expr."""
    from narwhals.expr import Expr

    return isinstance(obj, Expr)


def combine_evaluate_output_names(
    *exprs: CompliantExpr[CompliantFrameT, Any],
) -> Callable[[CompliantFrameT], Sequence[str]]:
    # Follow left-hand-rule for naming. E.g. `nw.sum_horizontal(expr1, expr2)` takes the
    # first name of `expr1`.
    if not is_compliant_expr(exprs[0]):  # pragma: no cover
        msg = f"Safety assertion failed, expected expression, got: {type(exprs[0])}. Please report a bug."
        raise AssertionError(msg)

    def evaluate_output_names(df: CompliantFrameT) -> Sequence[str]:
        return exprs[0]._evaluate_output_names(df)[:1]

    return evaluate_output_names


def combine_alias_output_names(
    *exprs: CompliantExpr[Any, Any],
) -> Callable[[Sequence[str]], Sequence[str]] | None:
    # Follow left-hand-rule for naming. E.g. `nw.sum_horizontal(expr1.alias(alias), expr2)` takes the
    # aliasing function of `expr1` and apply it to the first output name of `expr1`.
    if exprs[0]._alias_output_names is None:
        return None

    def alias_output_names(names: Sequence[str]) -> Sequence[str]:
        return exprs[0]._alias_output_names(names)[:1]  # type: ignore[misc]

    return alias_output_names


def extract_compliant(
    plx: CompliantNamespace[Any, CompliantExprT], other: Any, *, str_as_lit: bool
) -> CompliantExprT | object:
    if is_expr(other):
        return other._to_compliant_expr(plx)
    if isinstance(other, str) and not str_as_lit:
        return plx.col(other)
    if is_narwhals_series(other):
        return other._compliant_series._to_expr()
    if is_numpy_array(other):
        ns = cast("EagerNamespaceAny", plx)
        return ns._series.from_numpy(other, context=ns)._to_expr()
    return other


def evaluate_output_names_and_aliases(
    expr: CompliantExpr[Any, Any],
    df: CompliantDataFrame[Any, Any, Any] | CompliantLazyFrame[Any, Any],
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
    if exclude:
        assert expr._metadata is not None  # noqa: S101
        if expr._metadata.expansion_kind.is_multi_unnamed():
            output_names, aliases = zip(
                *[
                    (x, alias)
                    for x, alias in zip(output_names, aliases)
                    if x not in exclude
                ]
            )
    return output_names, aliases


class ExprKind(Enum):
    """Describe which kind of expression we are dealing with.

    Commutative composition rules are:
    - LITERAL vs LITERAL -> LITERAL
    - FILTRATION vs (LITERAL | AGGREGATION) -> FILTRATION
    - FILTRATION vs (FILTRATION | TRANSFORM | WINDOW) -> raise
    - (TRANSFORM | WINDOW) vs (LITERAL | AGGREGATION) -> TRANSFORM
    - AGGREGATION vs (LITERAL | AGGREGATION) -> AGGREGATION
    """

    LITERAL = auto()
    """e.g. `nw.lit(1)`"""

    AGGREGATION = auto()
    """e.g. `nw.col('a').mean()`"""

    TRANSFORM = auto()
    """preserves length, e.g. `nw.col('a').round()`"""

    WINDOW = auto()
    """transform in which last node is order-dependent

    examples:
    - `nw.col('a').cum_sum()`
    - `(nw.col('a')+1).cum_sum()`

    non-examples:
    - `nw.col('a').cum_sum()+1`
    - `nw.col('a').cum_sum().mean()`
    """

    FILTRATION = auto()
    """e.g. `nw.col('a').drop_nulls()`"""

    def preserves_length(self) -> bool:
        return self in {ExprKind.TRANSFORM, ExprKind.WINDOW}

    def is_window(self) -> bool:
        return self is ExprKind.WINDOW

    def is_filtration(self) -> bool:
        return self is ExprKind.FILTRATION

    def is_scalar_like(self) -> bool:
        return is_scalar_like(self)


def is_scalar_like(
    kind: ExprKind,
) -> TypeIs[Literal[ExprKind.AGGREGATION, ExprKind.LITERAL]]:
    # Like ExprKind.is_scalar_like, but uses TypeIs for better type checking.
    return kind in {ExprKind.AGGREGATION, ExprKind.LITERAL}


class ExpansionKind(Enum):
    """Describe what kind of expansion the expression performs."""

    SINGLE = auto()
    """e.g. `nw.col('a'), nw.sum_horizontal(nw.all())`"""

    MULTI_NAMED = auto()
    """e.g. `nw.col('a', 'b')`"""

    MULTI_UNNAMED = auto()
    """e.g. `nw.all()`, nw.nth(0, 1)"""

    def is_multi_unnamed(self) -> bool:
        return self is ExpansionKind.MULTI_UNNAMED


def is_multi_output(
    expansion_kind: ExpansionKind,
) -> TypeIs[Literal[ExpansionKind.MULTI_NAMED, ExpansionKind.MULTI_UNNAMED]]:
    return expansion_kind in {ExpansionKind.MULTI_NAMED, ExpansionKind.MULTI_UNNAMED}


class ExprMetadata:
    __slots__ = ("_expansion_kind", "_kind", "_n_open_windows")

    def __init__(
        self, kind: ExprKind, /, *, n_open_windows: int, expansion_kind: ExpansionKind
    ) -> None:
        self._kind: ExprKind = kind
        self._n_open_windows = n_open_windows
        self._expansion_kind = expansion_kind

    def __init_subclass__(cls, /, *args: Any, **kwds: Any) -> Never:  # pragma: no cover
        msg = f"Cannot subclass {cls.__name__!r}"
        raise TypeError(msg)

    def __repr__(self) -> str:
        return f"ExprMetadata(kind: {self._kind}, n_open_windows: {self._n_open_windows}, expansion_kind: {self._expansion_kind})"

    @property
    def kind(self) -> ExprKind:
        return self._kind

    @property
    def n_open_windows(self) -> int:
        return self._n_open_windows

    @property
    def expansion_kind(self) -> ExpansionKind:
        return self._expansion_kind

    def with_kind(self, kind: ExprKind, /) -> ExprMetadata:
        """Change metadata kind, leaving all other attributes the same."""
        return ExprMetadata(
            kind, n_open_windows=self._n_open_windows, expansion_kind=self._expansion_kind
        )

    def with_extra_open_window(self) -> ExprMetadata:
        """Increment `n_open_windows` leaving other attributes the same."""
        return ExprMetadata(
            self.kind,
            n_open_windows=self._n_open_windows + 1,
            expansion_kind=self._expansion_kind,
        )

    def with_kind_and_extra_open_window(self, kind: ExprKind, /) -> ExprMetadata:
        """Change metadata kind and increment `n_open_windows`."""
        return ExprMetadata(
            kind,
            n_open_windows=self._n_open_windows + 1,
            expansion_kind=self._expansion_kind,
        )

    @staticmethod
    def simple_selector() -> ExprMetadata:
        # e.g. `nw.col('a')`, `nw.nth(0)`
        return ExprMetadata(
            ExprKind.TRANSFORM, n_open_windows=0, expansion_kind=ExpansionKind.SINGLE
        )

    @staticmethod
    def multi_output_selector_named() -> ExprMetadata:
        # e.g. `nw.col('a', 'b')`
        return ExprMetadata(
            ExprKind.TRANSFORM, n_open_windows=0, expansion_kind=ExpansionKind.MULTI_NAMED
        )

    @staticmethod
    def multi_output_selector_unnamed() -> ExprMetadata:
        # e.g. `nw.all()`
        return ExprMetadata(
            ExprKind.TRANSFORM,
            n_open_windows=0,
            expansion_kind=ExpansionKind.MULTI_UNNAMED,
        )


def combine_metadata(
    *args: IntoExpr | object | None,
    str_as_lit: bool,
    allow_multi_output: bool,
    to_single_output: bool,
) -> ExprMetadata:
    """Combine metadata from `args`.

    Arguments:
        args: Arguments, maybe expressions, literals, or Series.
        str_as_lit: Whether to interpret strings as literals or as column names.
        allow_multi_output: Whether to allow multi-output inputs.
        to_single_output: Whether the result is always single-output, regardless
            of the inputs (e.g. `nw.sum_horizontal`).
    """
    n_filtrations = 0
    has_transforms_or_windows = False
    has_aggregations = False
    has_literals = False
    result_n_open_windows = 0
    result_expansion_kind = ExpansionKind.SINGLE

    for i, arg in enumerate(args):
        if isinstance(arg, str) and not str_as_lit:
            has_transforms_or_windows = True
        elif is_expr(arg):
            if is_multi_output(arg._metadata.expansion_kind):
                if i > 0 and not allow_multi_output:
                    # Left-most argument is always allowed to be multi-output.
                    msg = (
                        "Multi-output expressions (e.g. nw.col('a', 'b'), nw.all()) "
                        "are not supported in this context."
                    )
                    raise MultiOutputExpressionError(msg)
                if not to_single_output:
                    if i == 0:
                        result_expansion_kind = arg._metadata.expansion_kind
                    else:
                        result_expansion_kind = resolve_expansion_kind(
                            result_expansion_kind, arg._metadata.expansion_kind
                        )
            if arg._metadata.n_open_windows:
                result_n_open_windows += 1
            kind = arg._metadata.kind
            if kind is ExprKind.AGGREGATION:
                has_aggregations = True
            elif kind is ExprKind.LITERAL:
                has_literals = True
            elif kind is ExprKind.FILTRATION:
                n_filtrations += 1
            elif kind.preserves_length():
                has_transforms_or_windows = True
            else:  # pragma: no cover
                msg = "unreachable code"
                raise AssertionError(msg)

    if (
        has_literals
        and not has_aggregations
        and not has_transforms_or_windows
        and not n_filtrations
    ):
        result_kind = ExprKind.LITERAL
    elif n_filtrations > 1:
        msg = "Length-changing expressions can only be used in isolation, or followed by an aggregation"
        raise LengthChangingExprError(msg)
    elif n_filtrations and has_transforms_or_windows:
        msg = "Cannot combine length-changing expressions with length-preserving ones or aggregations"
        raise ShapeError(msg)
    elif n_filtrations:
        result_kind = ExprKind.FILTRATION
    elif has_transforms_or_windows:
        result_kind = ExprKind.TRANSFORM
    else:
        result_kind = ExprKind.AGGREGATION

    return ExprMetadata(
        result_kind,
        n_open_windows=result_n_open_windows,
        expansion_kind=result_expansion_kind,
    )


def resolve_expansion_kind(lhs: ExpansionKind, rhs: ExpansionKind) -> ExpansionKind:
    if lhs is ExpansionKind.MULTI_UNNAMED and rhs is ExpansionKind.MULTI_UNNAMED:
        # e.g. nw.selectors.all() - nw.selectors.numeric().
        return ExpansionKind.MULTI_UNNAMED
    # Don't attempt anything more complex, keep it simple and raise in the face of ambiguity.
    msg = f"Unsupported ExpansionKind combination, got {lhs} and {rhs}, please report a bug."  # pragma: no cover
    raise AssertionError(msg)  # pragma: no cover


def combine_metadata_binary_op(lhs: Expr, rhs: IntoExpr) -> ExprMetadata:
    # We may be able to allow multi-output rhs in the future:
    # https://github.com/narwhals-dev/narwhals/issues/2244.
    return combine_metadata(
        lhs, rhs, str_as_lit=True, allow_multi_output=False, to_single_output=False
    )


def combine_metadata_horizontal_op(*exprs: IntoExpr) -> ExprMetadata:
    return combine_metadata(
        *exprs, str_as_lit=False, allow_multi_output=True, to_single_output=True
    )


def check_expressions_preserve_length(*args: IntoExpr, function_name: str) -> None:
    # Raise if any argument in `args` isn't length-preserving.
    # For Series input, we don't raise (yet), we let such checks happen later,
    # as this function works lazily and so can't evaluate lengths.
    from narwhals.series import Series

    if not all(
        (is_expr(x) and x._metadata.kind.preserves_length())
        or isinstance(x, (str, Series))
        for x in args
    ):
        msg = f"Expressions which aggregate or change length cannot be passed to '{function_name}'."
        raise ShapeError(msg)


def all_exprs_are_scalar_like(*args: IntoExpr, **kwargs: IntoExpr) -> bool:
    # Raise if any argument in `args` isn't an aggregation or literal.
    # For Series input, we don't raise (yet), we let such checks happen later,
    # as this function works lazily and so can't evaluate lengths.
    exprs = chain(args, kwargs.values())
    return all(is_expr(x) and x._metadata.kind.is_scalar_like() for x in exprs)


def infer_kind(obj: IntoExpr | _1DArray | object, *, str_as_lit: bool) -> ExprKind:
    if is_expr(obj):
        return obj._metadata.kind
    if (
        is_narwhals_series(obj)
        or is_numpy_array(obj)
        or (isinstance(obj, str) and not str_as_lit)
    ):
        return ExprKind.TRANSFORM
    return ExprKind.LITERAL


def apply_n_ary_operation(
    plx: CompliantNamespace[Any, Any],
    function: Any,
    *comparands: IntoExpr,
    str_as_lit: bool,
) -> CompliantExpr[Any, Any]:
    compliant_exprs = (
        extract_compliant(plx, comparand, str_as_lit=str_as_lit)
        for comparand in comparands
    )
    kinds = [infer_kind(comparand, str_as_lit=str_as_lit) for comparand in comparands]

    broadcast = any(kind.preserves_length() for kind in kinds)
    compliant_exprs = (
        compliant_expr.broadcast(kind)
        if broadcast and is_compliant_expr(compliant_expr) and is_scalar_like(kind)
        else compliant_expr
        for compliant_expr, kind in zip(compliant_exprs, kinds)
    )
    return function(*compliant_exprs)
