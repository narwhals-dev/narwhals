# Utilities for expression parsing
# Useful for backends which don't have any concept of expressions, such
# and pandas or PyArrow.
from __future__ import annotations

from enum import Enum
from enum import auto
from itertools import chain
from typing import TYPE_CHECKING
from typing import Any
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
    from narwhals._compliant import CompliantFrameT
    from narwhals._compliant.typing import AliasNames
    from narwhals._compliant.typing import CompliantExprAny
    from narwhals._compliant.typing import CompliantFrameAny
    from narwhals._compliant.typing import CompliantNamespaceAny
    from narwhals._compliant.typing import EagerNamespaceAny
    from narwhals._compliant.typing import EvalNames
    from narwhals.expr import Expr
    from narwhals.typing import IntoExpr
    from narwhals.typing import NonNestedLiteral
    from narwhals.typing import _1DArray

    T = TypeVar("T")


def is_expr(obj: Any) -> TypeIs[Expr]:
    """Check whether `obj` is a Narwhals Expr."""
    from narwhals.expr import Expr

    return isinstance(obj, Expr)


def combine_evaluate_output_names(
    *exprs: CompliantExpr[CompliantFrameT, Any],
) -> EvalNames[CompliantFrameT]:
    # Follow left-hand-rule for naming. E.g. `nw.sum_horizontal(expr1, expr2)` takes the
    # first name of `expr1`.
    if not is_compliant_expr(exprs[0]):  # pragma: no cover
        msg = f"Safety assertion failed, expected expression, got: {type(exprs[0])}. Please report a bug."
        raise AssertionError(msg)

    def evaluate_output_names(df: CompliantFrameT) -> Sequence[str]:
        return exprs[0]._evaluate_output_names(df)[:1]

    return evaluate_output_names


def combine_alias_output_names(*exprs: CompliantExprAny) -> AliasNames | None:
    # Follow left-hand-rule for naming. E.g. `nw.sum_horizontal(expr1.alias(alias), expr2)` takes the
    # aliasing function of `expr1` and apply it to the first output name of `expr1`.
    if exprs[0]._alias_output_names is None:
        return None

    def alias_output_names(names: Sequence[str]) -> Sequence[str]:
        return exprs[0]._alias_output_names(names)[:1]  # type: ignore[misc]

    return alias_output_names


def extract_compliant(
    plx: CompliantNamespaceAny,
    other: IntoExpr | NonNestedLiteral | _1DArray,
    *,
    str_as_lit: bool,
) -> CompliantExprAny | NonNestedLiteral:
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
    expr: CompliantExprAny, df: CompliantFrameAny, exclude: Sequence[str]
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
    - (TRANSFORM | WINDOW) vs (...) -> TRANSFORM
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


class WindowKind(Enum):
    """Describe what kind of window the expression contains."""

    NONE = auto()
    """e.g. `nw.col('a').abs()`, no windows."""

    CLOSEABLE = auto()
    """e.g. `nw.col('a').cum_sum()` - can be closed if immediately followed by `over(order_by=...)`."""

    UNCLOSEABLE = auto()
    """e.g. `nw.col('a').cum_sum().abs()` - the window function (`cum_sum`) wasn't immediately followed by
    `over(order_by=...)`, and so the window is uncloseable.

    Uncloseable windows can be used freely in `nw.DataFrame`, but not in `nw.LazyFrame` where
    row-order is undefined."""

    CLOSED = auto()
    """e.g. `nw.col('a').cum_sum().over(order_by='i')`."""

    def is_open(self) -> bool:
        return self in {WindowKind.UNCLOSEABLE, WindowKind.CLOSEABLE}

    def is_closed(self) -> bool:
        return self is WindowKind.CLOSED

    def is_uncloseable(self) -> bool:
        return self is WindowKind.UNCLOSEABLE


class ExprMetadata:
    __slots__ = ("_expansion_kind", "_kind", "_window_kind")

    def __init__(
        self,
        kind: ExprKind,
        /,
        *,
        window_kind: WindowKind,
        expansion_kind: ExpansionKind,
    ) -> None:
        self._kind: ExprKind = kind
        self._window_kind = window_kind
        self._expansion_kind = expansion_kind

    def __init_subclass__(cls, /, *args: Any, **kwds: Any) -> Never:  # pragma: no cover
        msg = f"Cannot subclass {cls.__name__!r}"
        raise TypeError(msg)

    def __repr__(self) -> str:
        return f"ExprMetadata(kind: {self._kind}, window_kind: {self._window_kind}, expansion_kind: {self._expansion_kind})"

    @property
    def kind(self) -> ExprKind:
        return self._kind

    @property
    def window_kind(self) -> WindowKind:
        return self._window_kind

    @property
    def expansion_kind(self) -> ExpansionKind:
        return self._expansion_kind

    def with_kind(self, kind: ExprKind, /) -> ExprMetadata:
        """Change metadata kind, leaving all other attributes the same."""
        return ExprMetadata(
            kind,
            window_kind=self._window_kind,
            expansion_kind=self._expansion_kind,
        )

    def with_uncloseable_window(self) -> ExprMetadata:
        """Add uncloseable window, leaving other attributes the same."""
        if self._window_kind is WindowKind.CLOSED:  # pragma: no cover
            msg = "Unreachable code, please report a bug."
            raise AssertionError(msg)
        return ExprMetadata(
            self.kind,
            window_kind=WindowKind.UNCLOSEABLE,
            expansion_kind=self._expansion_kind,
        )

    def with_kind_and_closeable_window(self, kind: ExprKind, /) -> ExprMetadata:
        """Change metadata kind and add closeable window.

        If we already have an uncloseable window, the window stays uncloseable.
        """
        if self._window_kind is WindowKind.NONE:
            window_kind = WindowKind.CLOSEABLE
        elif self._window_kind is WindowKind.CLOSED:  # pragma: no cover
            msg = "Unreachable code, please report a bug."
            raise AssertionError(msg)
        else:
            window_kind = WindowKind.UNCLOSEABLE
        return ExprMetadata(
            kind,
            window_kind=window_kind,
            expansion_kind=self._expansion_kind,
        )

    def with_kind_and_uncloseable_window(self, kind: ExprKind, /) -> ExprMetadata:
        """Change metadata kind and set window kind to uncloseable."""
        return ExprMetadata(
            kind,
            window_kind=WindowKind.UNCLOSEABLE,
            expansion_kind=self._expansion_kind,
        )

    @staticmethod
    def selector_single() -> ExprMetadata:
        # e.g. `nw.col('a')`, `nw.nth(0)`
        return ExprMetadata(
            ExprKind.TRANSFORM,
            window_kind=WindowKind.NONE,
            expansion_kind=ExpansionKind.SINGLE,
        )

    @staticmethod
    def selector_multi_named() -> ExprMetadata:
        # e.g. `nw.col('a', 'b')`
        return ExprMetadata(
            ExprKind.TRANSFORM,
            window_kind=WindowKind.NONE,
            expansion_kind=ExpansionKind.MULTI_NAMED,
        )

    @staticmethod
    def selector_multi_unnamed() -> ExprMetadata:
        # e.g. `nw.all()`
        return ExprMetadata(
            ExprKind.TRANSFORM,
            window_kind=WindowKind.NONE,
            expansion_kind=ExpansionKind.MULTI_UNNAMED,
        )


def combine_metadata(  # noqa: PLR0915
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
    result_expansion_kind = ExpansionKind.SINGLE
    has_closeable_windows = False
    has_uncloseable_windows = False
    has_closed_windows = False

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

            window_kind = arg._metadata.window_kind
            if window_kind is WindowKind.UNCLOSEABLE:
                has_uncloseable_windows = True
            elif window_kind is WindowKind.CLOSEABLE:
                has_closeable_windows = True
            elif window_kind is WindowKind.CLOSED:
                has_closed_windows = True

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

    if has_uncloseable_windows or has_closeable_windows:
        result_window_kind = WindowKind.UNCLOSEABLE
    elif has_closed_windows:
        result_window_kind = WindowKind.CLOSED
    else:
        result_window_kind = WindowKind.NONE

    return ExprMetadata(
        result_kind, window_kind=result_window_kind, expansion_kind=result_expansion_kind
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


def infer_kind(
    obj: IntoExpr | NonNestedLiteral | _1DArray, *, str_as_lit: bool
) -> ExprKind:
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
    plx: CompliantNamespaceAny,
    function: Any,
    *comparands: IntoExpr | NonNestedLiteral | _1DArray,
    str_as_lit: bool,
) -> CompliantExprAny:
    compliant_exprs = (
        extract_compliant(plx, comparand, str_as_lit=str_as_lit)
        for comparand in comparands
    )
    kinds = [infer_kind(comparand, str_as_lit=str_as_lit) for comparand in comparands]

    broadcast = any(not kind.is_scalar_like() for kind in kinds)
    compliant_exprs = (
        compliant_expr.broadcast(kind)
        if broadcast and is_compliant_expr(compliant_expr) and is_scalar_like(kind)
        else compliant_expr
        for compliant_expr, kind in zip(compliant_exprs, kinds)
    )
    return function(*compliant_exprs)
