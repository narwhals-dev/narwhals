from __future__ import annotations

try:
    import altair as alt
except ImportError as err:
    msg = "`altair` is required to convert `ExprIR`s to vega expressions."
    raise ModuleNotFoundError(msg) from err

import datetime as dt
import functools
import math
import operator
from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING, Any, Final, Literal, TypeAlias, TypeVar, final

from altair.expr import core as alt_ir

from narwhals._plan import _function, expressions as ir
from narwhals._plan.altair.api._parameter_ir import _ParameterIR
from narwhals._plan.altair.api.exceptions import unsupported_error
from narwhals._plan.altair.api.typing import AltExpr, IntoAltExpr
from narwhals._plan.expr import Expr as NwExpr
from narwhals._plan.expressions import (
    aggregation as agg,
    boolean,
    functions as F,
    operators as ops,
    strings,
    temporal,
)

if TYPE_CHECKING:
    from altair.vegalite.v6.schema._typing import SingleTimeUnit_T

    from narwhals._plan.altair.api.typing import VegaExpr
    from narwhals._plan.typing import Seq
    from narwhals.typing import PythonLiteral


_VT_co = TypeVar("_VT_co", covariant=True)
_T = TypeVar("_T")

AltFnName: TypeAlias = Literal[
    "if", "max", "min", "indexof", "inrange", "peek", "length", "sort"
]

FnMap: TypeAlias = Mapping[type[ir.Function], _VT_co]
AltUnary: TypeAlias = Callable[[AltExpr], AltExpr]
AltBinary: TypeAlias = Callable[[AltExpr, AltExpr], AltExpr]

_COMMA = ","


@final
class AltExprStr(alt_ir.Expression):
    """Extensions to altair's expressions.

    ## Notes
    - pre-bakes the string conversion
        - more likely that you'll write something complex
    - TODO @dangotbanned: investigate not inheriting `Expression` (and `SchemaBase`)
        - I just want a thin wrapper around a string and to hash it
    """

    js_repr: VegaExpr

    def __init__(self, js_repr: VegaExpr, /) -> None:
        super().__init__(js_repr=js_repr)

    def __repr__(self) -> VegaExpr:
        return self.js_repr

    __str__ = __repr__

    def __copy__(self) -> AltExprStr:
        return self

    def __deepcopy__(self, memo: Any) -> AltExprStr:
        return self

    @staticmethod
    def list_expr(exprs: Iterable[alt_ir.Expression], /) -> AltExprStr:
        """Create a JS array literal from expressions.

        https://github.com/vega/altair/issues/3605
        """
        return AltExprStr(_list_fmt(exprs, repr))

    @staticmethod
    def call_fn(function: AltFnName, exprs: Iterable[alt_ir.Expression], /) -> AltExprStr:
        return AltExprStr(f"{function}({_COMMA.join(repr(e) for e in exprs)})")

    @staticmethod
    def call_fn_unary(function: str, expr: alt_ir.Expression, /) -> AltExprStr:
        return AltExprStr(f"{function}({expr!r})")

    @staticmethod
    def get_item(
        owner: Literal["datum"] | AltExprStr | str,  # noqa: PYI051
        item: int | str,
        /,
    ) -> AltExprStr:
        # sooooo overloaded
        # - `owner = "datum"`
        #   - `item` is a column name
        # - `owner = <expr-result>`
        #   - `item: str`
        #     - attribute access, with escaping
        #   - `item: int`
        #     - "array" element access, where that means the row index
        return AltExprStr(f"{owner}[{item!r}]")


def _list_fmt(values: Iterable[_T], to_str: Callable[[_T], str], /) -> VegaExpr:
    return f"[{_COMMA.join(to_str(value) for value in values)}]"


def _dict_fmt(
    items: Iterable[tuple[str, _T]], to_str: Callable[[_T], str], /
) -> VegaExpr:
    lb, rb = "{", "}"
    return f"{lb}{_COMMA.join(f'{k!r}:{to_str(v)}' for k, v in items)}{rb}"


# NOTE: classmethod accessor
ae = alt.expr


def into_vega_expr(expr: ir.ExprIR, /) -> VegaExpr:
    """Convert an [`ExprIR`][narwhals._plan.expressions.ExprIR] into a [Vega Expression].

    [Vega Expression]: https://vega.github.io/vega/docs/expressions/

    Arguments:
        expr: The intermediate representation of an [`Expr`][narwhals._plan.Expr].

    Raises:
        NotImplementedError: If the expression contains any unsupported operations.

    ## Notes
    Common unsupported expressions:

    - [`Alias`][narwhals._plan.expressions.Alias]
        - If the outer context supports it, use `parse_into_named_exprs` *first* to resolve the output name
    - [`SelectorIR`][narwhals._plan.expressions.SelectorIR]
        - (Vega) expressions and transforms can produce new "columns"
          but they do not change the schema of the original data source.
    - Most aggregations & non-elementwise functions
        - Many *can* be translated, but only for specific contexts like `Chart.transform_*` or `Chart.encode`
    """
    # NOTE: Can safely add `@functools.lru_cache` here, or on any
    # function in this module that accepts only `ExprIR` or `Function`
    return repr(_from_expr_ir(expr))


@functools.singledispatch
def _from_expr_ir(expr: ir.ExprIR) -> AltExpr:
    raise unsupported_error(expr, "expr")


@_from_expr_ir.register(ir.Column)
def _(expr: ir.Column) -> AltExprStr:
    return AltExprStr.get_item("datum", expr.name)


@_from_expr_ir.register(ir.BinaryExpr)
def _(expr: ir.BinaryExpr) -> AltExpr:
    lhs, rhs = _from_expr_ir(expr.left), _from_expr_ir(expr.right)
    return _from_binary(expr.op, lhs, rhs)


@_from_expr_ir.register(ir.TernaryExpr)
def _(expr: ir.TernaryExpr) -> AltExpr:
    exprs = (_from_expr_ir(e) for e in (expr.predicate, expr.truthy, expr.falsy))
    return AltExprStr.call_fn("if", exprs)


@_from_expr_ir.register(_ParameterIR)
def _(expr: _ParameterIR) -> AltExpr:
    return AltExprStr(expr.name)


def parse_into_alt_expr(
    expr: NwExpr | VegaExpr | alt.ExprRef | alt_ir.OperatorMixin | alt.Expr, /
) -> AltExpr:
    """Convert almost anything into a composable altair expression.

    Convenience wrapper around [`into_vega_expr`][].

    Arguments:
        expr: An [`Expr`][narwhals._plan.Expr] or anything that is
            currently accepted as an altair expression/predicate.
    """
    return _into_alt_expr(expr)


def _identity_opaque(obj: Any, /) -> Any:
    # Using an unbound type param is more accurate, but interacts badly with singledispatch
    return obj


@functools.singledispatch
def _into_alt_expr(expr: Any, /) -> AltExpr:
    msg = f"Cannot convert an {type(expr).__name__!r} a vega expression, got: {expr!r}"
    raise TypeError(msg)


@_into_alt_expr.register(NwExpr)
def _(expr: NwExpr, /) -> AltExpr:
    return _from_expr_ir(expr._ir)


_into_alt_expr.register(str, AltExprStr)
_into_alt_expr.register(alt.Expr, AltExprStr)
_into_alt_expr.register(alt_ir.Expression, _identity_opaque)
_into_alt_expr.register(alt.Parameter, lambda p: _into_alt_expr(p.name))
# rewraps `alt.expr("...")` -> `AltExprStr("...")`
_into_alt_expr.register(alt.ExprRef, lambda e: _into_alt_expr(e.expr))
_into_alt_expr.register(alt_ir.OperatorMixin, lambda op: _into_alt_expr(op._to_expr()))


def parse_into_vega_expr(expr: NwExpr | IntoAltExpr, /) -> VegaExpr:
    """Convenience wrapper around [`into_vega_expr`][], returning a string.

    Arguments:
        expr: An [`Expr`][narwhals._plan.Expr] or anything that is
            currently accepted as an altair expression.
    """
    return _into_vega_expr(expr)


@functools.singledispatch
def _into_vega_expr(expr: NwExpr | IntoAltExpr, /) -> VegaExpr:
    msg = f"Cannot convert an {type(expr).__name__!r} a vega expression, got: {expr!r}"
    raise TypeError(msg)


@_into_vega_expr.register(NwExpr)
def _(expr: NwExpr, /) -> VegaExpr:
    return into_vega_expr(expr._ir)


_into_vega_expr.register(str, _identity_opaque)
_into_vega_expr.register(alt_ir.Expression, repr)


# TODO @dangotbanned: misc
# - defined for OperatorMixin
#   - Pow (https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Operators/Exponentiation)
@_from_expr_ir.register(ir.FunctionExpr)
def _function_expr(expr: ir.FunctionExpr) -> AltExpr:
    result = _from_function(expr.function, expr.args)
    if result is not None:
        return result
    raise unsupported_error(expr, "expr")


@functools.singledispatch
def _from_function(f: ir.Function, args: Seq[ir.ExprIR], /) -> AltExpr | None:  # noqa: ARG001
    """Return `None` when we can't translate (yet?).

    The caller (`_function_expr`) can raise with the full `ExprIR`
    to use in an error message.
    """
    return None


@functools.singledispatch
def _from_function_horizontal(
    f: _function.Horizontal,  # noqa: ARG001
    exprs: Iterable[AltExpr],  # noqa: ARG001
    /,
) -> AltExpr | None:
    return None


def _is_null(expr: AltExpr, /) -> AltExpr:
    # NOTE: ... because javascript 🙃
    # https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Operators/Strict_equality
    # this is technically valid for python, but is meaningless in `polars`
    # and you'll get a warning saying to use `is_null`
    result: AltExpr = expr == None  # noqa: E711
    return result


def _then_invert(func: AltUnary, /) -> AltUnary:
    def _(expr: AltExpr, /) -> AltExpr:
        result: AltExpr = ~func(expr)
        return result

    return _


def _rewrite_is_between(f: boolean.IsBetween, args: Seq[ir.ExprIR], /) -> AltExpr:
    """Similar to `FieldRangePredicate`.

    Adapted from [vega-lite](https://github.com/vega/vega-lite/blob/91845a3bca89f9e8bd6ae847bc1b3f31cc85e919/src/predicate.ts#L235-L242)

    Note:
        Not handling the null cases, as I found no tests in narwhals/polars that expect that to work
    """
    if f.closed != "both":
        expr_ir = f.to_function_expr(*args)
        raise unsupported_error(expr_ir, "expr", "non-default")
    root = args[0]
    prev, lower, upper = (_from_expr_ir(arg) for arg in args)
    if isinstance(root, ir.FunctionExpr) and (
        time_unit := _TEMPORAL_TIME_UNIT.get(type(root.function))
    ):
        return AltExprStr(
            f"inrange({prev!r}, [{time_unit}({lower!r}), {time_unit}({upper!r})])"
        )
    # NOTE: 2-tuples as arguments, great
    return AltExprStr(f"inrange({prev!r}, [{lower!r}, {upper!r}])")


def _rewrite_is_in_seq(
    f: boolean.IsInSeq, args: Seq[ir.ExprIR], /, *, is_not_in: bool = False
) -> AltExpr:
    """Similar to `FieldOneOfPredicate`.

    Adapted from [vega-lite](https://github.com/vega/vega-lite/blob/91845a3bca89f9e8bd6ae847bc1b3f31cc85e919/src/predicate.ts#L231-L232)

    Can use instead of chaining `__or__`:

        (alt.datum.label == "Begin") | (alt.datum.label == "Middle") | (alt.datum.label == "End")
        nw.col("label").is_in(["Begin", "Middle", "End"])
    """
    # NOTE: Seems hacky, but prefixing `!` doesn't seem to work?
    op = "===" if is_not_in else "!=="
    return AltExprStr(f"indexof({_from_lit(f.other)},{_from_expr_ir(args[0])!r}) {op} -1")


def _rewrite_struct_field_getitem(
    f: ir.struct.FieldByName, args: Seq[ir.ExprIR], /
) -> AltExpr:
    # https://github.com/vega/altair/blob/7b289f9adf07d960f94620ef02c33b8f400f4948/tests/examples_methods_syntax/isotype_emoji.py#L58-L59
    return AltExprStr.get_item(repr(_from_expr_ir(args[0])), f.name)


def _rewrite_not(_: boolean.Not, args: Seq[ir.ExprIR], /) -> AltExpr:
    prev = args[0]
    if isinstance(prev, ir.FunctionExpr) and isinstance(prev.function, boolean.IsInSeq):
        return _rewrite_is_in_seq(prev.function, prev.args, is_not_in=True)
    return AltExprStr(f"(!{_from_expr_ir(prev)!r})")


_UNARY_BOOLEAN: FnMap[AltUnary] = {
    boolean.IsNull: _is_null,
    boolean.IsNotNull: _then_invert(_is_null),
    boolean.IsNan: ae.isNaN,
    boolean.IsNotNan: _then_invert(ae.isNaN),
    boolean.IsFinite: ae.isFinite,
}
# all the math stuff that works unconditionally
_UNARY_SIMPLE: FnMap[AltUnary] = {
    F.Abs: ae.abs,
    F.Ceil: ae.ceil,
    F.Exp: ae.exp,
    F.Floor: ae.floor,
    F.Sqrt: ae.sqrt,
}

_HORIZONTAL_REDUCE: FnMap[AltBinary] = {
    boolean.AnyHorizontal: operator.or_,
    boolean.AllHorizontal: operator.and_,
    F.SumHorizontal: operator.add,
}
_HORIZONTAL_NATIVE_NAME: FnMap[AltFnName] = {
    F.MaxHorizontal: "max",
    F.MinHorizontal: "min",
}

_REWRITE_FUNCTION: Final[FnMap[Callable[..., AltExpr]]] = {
    boolean.IsBetween: _rewrite_is_between,
    boolean.IsInSeq: _rewrite_is_in_seq,
    boolean.Not: _rewrite_not,
    ir.struct.FieldByName: _rewrite_struct_field_getitem,
}
"""Translations that do something fancy with the `Function`.

Particularly when what they do is a one-off thing.
"""

# https://github.com/vega/vega/blob/9fef6ea66a02973adc90f6824558c76996a5ef86/packages/vega-util/src/peek.ts
_UNARY_EXPR_FN_NAME: Mapping[type[ir.ExprIR], AltFnName] = {
    agg.Last: "peek",
    agg.Len: "length",
}
"""Similar to `FunctionExpr[Unary]`, but the expression comes from `self.expr`."""

_TEMPORAL_TIME_UNIT: FnMap[SingleTimeUnit_T] = {
    temporal.Year: "year",
    # has no equivalent
    # - "quarter" -> (Q1-Q4) polars has is without the prefix
    # - "month" -> (Jan-Dec)
    temporal.Day: "date",
    # has no equivalent
    # - "week" -> (W01-W52)
    # - "day" -> (Sunday-Saturday)
    temporal.OrdinalDay: "dayofyear",
    # NOTE: These aren't quite equivalent (vega -> Time, polars -> Int),
    # but for the purpose of visualization, you likely want the richer type
    temporal.Hour: "hours",
    temporal.Minute: "minutes",
    temporal.Second: "seconds",
    temporal.Millisecond: "milliseconds",
}

for _tp, _rewrite_fn in _REWRITE_FUNCTION.items():
    _from_function.register(_tp, _rewrite_fn)


def _boolean_fn(f: boolean.BooleanFunction, args: tuple[ir.ExprIR, ...], /) -> AltExpr:
    return _UNARY_BOOLEAN[type(f)](_from_expr_ir(args[0]))


for _tp in _UNARY_BOOLEAN:
    # NOTE: `BooleanFunction` is more derived in the mro than `Unary`
    # - `_HorizontalBoolean` is gobbled up by `HorizontalExpr`, so tis isn't obvious
    _from_function.register(_tp, _boolean_fn)


@_from_function.register(_function.Unary)
def _(f: _function.Unary, args: tuple[ir.ExprIR], /) -> AltExpr | None:
    expr = _from_expr_ir(args[0])
    if supported := _UNARY_SIMPLE.get(type(f)):
        return supported(expr)
    if time_unit := _TEMPORAL_TIME_UNIT.get(type(f)):
        return AltExprStr.call_fn_unary(time_unit, expr)
    match f:
        case F.Log(base=math.e):
            return ae.log(expr)
        case F.Round(decimals=0):
            return ae.round(expr)
        case F.Log() | F.Round():
            expr_ir = f.to_function_expr(*args)
            raise unsupported_error(expr_ir, "expr", "non-default")
        case strings._StringUnary() | ir.temporal._TemporalUnary():
            msg_0 = f"TODO: ({f.__expr_ir_dispatch__.options.accessor_name!r}, unary) {type(f).__name__!r}"
            raise NotImplementedError(msg_0)
        case _:
            return None


def _horizontal_reduce(f: _function.Horizontal, exprs: Iterable[AltExpr], /) -> AltExpr:
    return functools.reduce(_HORIZONTAL_REDUCE[type(f)], exprs)


for _tp in _HORIZONTAL_REDUCE:
    _from_function_horizontal.register(_tp, _horizontal_reduce)


def _horizontal_native(f: _function.Horizontal, exprs: Iterable[AltExpr], /) -> AltExpr:
    return AltExprStr.call_fn(_HORIZONTAL_NATIVE_NAME[type(f)], exprs)


for _tp in _HORIZONTAL_NATIVE_NAME:
    _from_function_horizontal.register(_tp, _horizontal_native)


@_from_function_horizontal.register(strings.ConcatStr)
def _rewrite_concat_str(f: strings.ConcatStr, exprs: Iterable[AltExpr], /) -> AltExpr:
    return ae.join(AltExprStr.list_expr(exprs), f.separator)


@_from_expr_ir.register(ir.HorizontalExpr)
def _(expr: ir.HorizontalExpr) -> AltExpr:
    args = (_from_expr_ir(arg) for arg in expr.args)
    result = _from_function_horizontal(expr.function, args)
    if result is not None:
        return result
    raise unsupported_error(expr, "expr")


@_from_expr_ir.register(agg.First)
def _(expr: agg.First) -> AltExprStr:
    return AltExprStr.get_item(repr(_from_expr_ir(expr.expr)), 0)


def _unary_expr_fn(expr: agg.Last | agg.Len) -> AltExprStr:
    prev = _from_expr_ir(expr.expr)
    return AltExprStr.call_fn_unary(_UNARY_EXPR_FN_NAME[type(expr)], prev)


for _tp in _UNARY_EXPR_FN_NAME:
    _from_expr_ir.register(_tp, _unary_expr_fn)


@_from_expr_ir.register(ir.Sort)
def _(expr: ir.Sort) -> AltExpr:
    # https://vega.github.io/vega/docs/expressions/#sort
    if expr.descending or expr.nulls_last:
        raise unsupported_error(expr, "expr", "non-default")
    return AltExprStr.call_fn_unary("sort", _from_expr_ir(expr.expr))


@_from_expr_ir.register(ir.Lit)
def _(expr: ir.Lit[PythonLiteral]) -> AltExpr:
    return AltExprStr(_from_lit(expr.value))


@functools.singledispatch
def _from_lit(value: PythonLiteral) -> VegaExpr:
    return repr(value)


_from_lit.register(dt.date, alt_ir._from_date_datetime)
_from_lit.register(type(None), lambda _: "null")
_from_lit.register(bool, lambda value: "true" if value else "false")


@_from_lit.register(list)
@_from_lit.register(tuple)
def _(value: list[Any] | tuple[Any, ...]) -> VegaExpr:
    """Adds support for `nw.lit([1,2,3])` -> `"[1,2,3]"`."""
    return _list_fmt(value, _from_lit)


@_from_lit.register(dict)
def _(value: dict[str, Any]) -> VegaExpr:
    """Adds support for `nw.lit({"a": 1, "b": 2})` -> `"{'a': 1, 'b': 2}"`."""
    return _dict_fmt(value.items(), _from_lit)


@functools.singledispatch
def _from_binary(op: ops.Operator, left: AltExpr, right: AltExpr) -> AltExpr:
    # re-uses altair's `BinaryExpression`, as it defines the JS operator differences
    result: AltExpr = op(left, right)
    return result


@_from_binary.register(ops.FloorDivide)
def _(op: ops.FloorDivide, left: AltExpr, right: AltExpr) -> AltExpr:
    msg = f"{ops.FloorDivide.__name__!r} is not a supported javascript operator"
    raise NotImplementedError(msg)


@_from_binary.register(ops.ExclusiveOr)
def _(op: ops.ExclusiveOr, left: AltExpr, right: AltExpr) -> alt_ir.BinaryExpression:  # noqa: ARG001
    result: alt_ir.BinaryExpression = (left & ~right) | (~left & right)
    return result
