from __future__ import annotations

from importlib.util import find_spec

from narwhals.typing import PythonLiteral

if find_spec("altair") is None:
    msg = "`altair` is required to convert `ExprIR`s"
    raise ModuleNotFoundError(msg)

from typing import TYPE_CHECKING, Literal, TypeAlias, TypedDict, Union

from altair.expr import core as alt_ir
from altair.vegalite.v6.schema import _typing as alt_t

if TYPE_CHECKING:
    from _typeshed import Incomplete
    from typing_extensions import NotRequired

    import narwhals._plan as nwp
    from narwhals._plan.typing import OneOrSeq


VegaExpr: TypeAlias = str
"""A stringized [Vega Expression].

[Vega Expression]: https://vega.github.io/vega/docs/expressions/
"""

AltExpr: TypeAlias = alt_ir.Expression
"""An altair wrapper around a [Vega Expression].

[Vega Expression]: https://vega.github.io/vega/docs/expressions/"""


Predicate: TypeAlias = VegaExpr | AltExpr
"""Anything [^1] that is accepted by `condition(test=...)`.

[^1]: That will be constructed via Narwhals expressions

See Also:
    [Predicate](https://vega.github.io/vega-lite/docs/predicate.html)
"""

FieldName: TypeAlias = str
"""The initial column name from `Col(name=...)`."""

OutputName: TypeAlias = str
"""The (optionally aliased) column name at the end of an expression.

Can be used for properties like:
    - `as`
    - `title`
"""

IntoExprColumn: TypeAlias = Union[FieldName, "nwp.Expr"]
"""Anything that can be converted into a column expression.

Many contexts will not support literals.
"""

IntoExpr: TypeAlias = IntoExprColumn | PythonLiteral
"""Anything that can be converted into an expression."""


AggregateOp: TypeAlias = alt_t.NonArgAggregateOp_T
"""Aggregates accepted by `WindowFieldDef(op=...)`.

Excludes `"argmin"`, `"argmax"` as they have a very different meaning to the polars methods.
"""

WindowOp: TypeAlias = alt_t.WindowOnlyOp_T
"""Window functions accepted by `WindowFieldDef(op=...)`."""

AggOrWindow: TypeAlias = AggregateOp | WindowOp
"""Anything accepted by `WindowFieldDef(op=...)`."""

_Repeater: TypeAlias = Literal["row", "column", "repeat", "layer"]


class ArgMin(TypedDict):
    argmin: FieldName


class ArgMax(TypedDict):
    argmax: FieldName


Aggregate: TypeAlias = AggregateOp | ArgMax | ArgMin
"""Anything accepted by `Encoding(aggregate=...)`.

## [`Arg{Max,Min}`](https://vega.github.io/vega-lite/docs/aggregate.html#argmax)

[`pl.Expr.{max,min}_by`]: https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.max_by.html#polars.Expr.max_by

These guys are pretty strange:

- No other aggregations in vega-lite accept arguments
    - Apparently you can use it without arguments in a `transform`
- The behavior in the example is equivalent to [`pl.Expr.{max,min}_by`]
- We don't have `nw.Expr.{max,min}_by`

*But*, we can use `sort_by` to match how max/min ignore nulls:

    alt.X("Production Budget", aggregate={"argmax": "US Gross"})
    pl.col("Production Budget").max_by("US Gross")
    nwp.col("Production Budget").sort_by("US Gross").last()

    alt.X("Production Budget", aggregate={"argmin": "US Gross"})
    pl.col("Production Budget").min_by("US Gross")
    nwp.col("Production Budget").sort_by("US Gross", nulls_last=True).first()
"""


TimeUnit: TypeAlias = "Incomplete"
"""TODO @dangotbanned: for `timeUnit`."""


class HasExpr(TypedDict):
    """`ExprRef`, which is anywhere `Parameter` is annotated."""

    expr: VegaExpr


class HasRepeat(TypedDict):
    """Tie a channel to the row or column within a repeated chart.

    Allowed in `Encoding(field=..., datum=...)`.
    """

    repeat: _Repeater


class HasType(TypedDict):
    # `"geojson"` is allowed in some places, but not all
    type: alt_t.StandardType_T
    """[`type`] has some interesting inference rules.

    [`type`]: https://vega.github.io/vega-lite/docs/type.html
    """


class MaybeTitle(TypedDict):
    title: NotRequired[OneOrSeq[str] | None]


class Datum(HasType, MaybeTitle, TypedDict):
    """Shared by all datums."""

    datum: HasExpr | alt_t.PrimitiveValue_T | HasRepeat
    """A constant data value.

    - Can never support transforms
    - But can support expressions
    """


class Field(HasType, MaybeTitle, TypedDict):
    """Shared by all fields."""

    # `HasRepeat` is always allowed, but not sure how it would make sense in narwhals
    field: FieldName  # | HasRepeat
    aggregate: NotRequired[Aggregate]
    timeUnit: NotRequired[TimeUnit]
    # On all fields, but havent mapped expressions over - maybe:
    # - hist?
    bin: NotRequired[Incomplete]


class Value(TypedDict):
    """Shared by all values."""

    value: HasExpr | alt_t.PrimitiveValue_T


AggField = TypedDict(
    "AggField", {"field": FieldName, "op": AggregateOp, "as": "NotRequired[FieldName]"}
)
WindowField = TypedDict(
    "WindowField",
    {
        "field": FieldName,
        "op": AggOrWindow,
        "as": "NotRequired[FieldName]",
        "param": "NotRequired[float]",
    },
)
