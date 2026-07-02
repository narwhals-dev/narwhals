from __future__ import annotations

from importlib.util import find_spec

from narwhals.typing import PythonLiteral

if find_spec("altair") is None:
    msg = "`altair` is required to convert `ExprIR`s"
    raise ModuleNotFoundError(msg)

import datetime as dt
from typing import TYPE_CHECKING, Literal, TypeAlias, Union

from altair.expr import core as alt_ir
from altair.typing import Optional as Optional  # noqa: PLC0414, TC002
from altair.vegalite.v6.schema import _typing as alt_t

if TYPE_CHECKING:
    import altair as alt
    from _typeshed import Incomplete
    from altair import typing as _alt_t
    from typing_extensions import NotRequired, TypedDict

    import narwhals._plan as nwp
    from narwhals._plan.typing import OneOrSeq
else:
    from typing import TypedDict

VegaType: TypeAlias = alt_t.StandardType_T

VegaExpr: TypeAlias = str
"""A stringized [Vega Expression].

[Vega Expression]: https://vega.github.io/vega/docs/expressions/
"""

AltExpr: TypeAlias = alt_ir.Expression
"""An altair wrapper around a [Vega Expression].

[Vega Expression]: https://vega.github.io/vega/docs/expressions/"""


IntoAltExpr: TypeAlias = VegaExpr | AltExpr
"""Anything that is currently accepted as an altair expression."""

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

NativeValue: TypeAlias = str | bool | float | dt.date | dt.datetime | None
"""Any python literal that is supported in `SchemaBase(value=...)`"""


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

Channel: TypeAlias = (
    alt_t.SingleDefUnitChannel_T
    | Literal[
        "column",
        "detail",
        "facet",
        "order",
        "row",
        "tooltip",
        "xError",
        "xError2",
        "yError",
        "yError2",
    ]
)
"""Keywords that can be passed to `Chart.encode(...)`."""


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


InnerValue: TypeAlias = NativeValue | HasExpr
"""Anything accepted by `Value(value=...)` or `Datum(datum=...)`."""


class HasRepeat(TypedDict):
    """Tie a channel to the row or column within a repeated chart.

    Allowed in `Encoding(field=..., datum=...)`.
    """

    repeat: _Repeater


class HasType(TypedDict):
    # `"geojson"` is allowed in some places, but not all
    type: NotRequired[Optional[VegaType]]
    """[`type`] has some interesting inference rules.

    [`type`]: https://vega.github.io/vega-lite/docs/type.html
    """


class MaybeTitle(TypedDict):
    title: NotRequired[OneOrSeq[str] | None]


class Datum(HasType, MaybeTitle, TypedDict):
    """Shared by all datums."""

    datum: InnerValue | HasRepeat
    """A constant data value.

    - Can never support transforms
    - But can support expressions
    """


class FieldOpen(HasType, MaybeTitle, TypedDict):
    """Shared by all fields."""

    # `HasRepeat` is always allowed, but not sure how it would make sense in narwhals
    field: FieldName  # | HasRepeat
    aggregate: Optional[Aggregate]
    timeUnit: NotRequired[TimeUnit]
    # On all fields, but havent mapped expressions over - maybe:
    # - hist?
    bin: NotRequired[Incomplete]


class ValueOpen(TypedDict):
    """Shared by all values."""

    value: InnerValue


if TYPE_CHECKING:
    # https://github.com/python/mypy/pull/21382
    class Field(FieldOpen, closed=True): ...  # type: ignore[call-arg]

    class Value(ValueOpen, closed=True): ...  # type: ignore[call-arg]
else:
    Field = FieldOpen
    Value = ValueOpen

# NOTE: `mypy` stops understanding any of the fields if `closed=True`
# is used with the functional syntax
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


class _EncodeKwds(TypedDict, total=False):
    """Encoding channels map properties of the data to visual properties of the chart."""

    angle: _alt_t.ChannelAngle | nwp.Expr
    color: _alt_t.ChannelColor | nwp.Expr
    column: _alt_t.ChannelColumn | nwp.Expr
    description: _alt_t.ChannelDescription | nwp.Expr
    detail: _alt_t.ChannelDetail | nwp.Expr
    facet: _alt_t.ChannelFacet | nwp.Expr
    fill: _alt_t.ChannelFill | nwp.Expr
    fillOpacity: _alt_t.ChannelFillOpacity | nwp.Expr
    href: _alt_t.ChannelHref | nwp.Expr
    key: _alt_t.ChannelKey | nwp.Expr
    latitude: _alt_t.ChannelLatitude | nwp.Expr
    latitude2: _alt_t.ChannelLatitude2 | nwp.Expr
    longitude: _alt_t.ChannelLongitude | nwp.Expr
    longitude2: _alt_t.ChannelLongitude2 | nwp.Expr
    opacity: _alt_t.ChannelOpacity | nwp.Expr
    order: _alt_t.ChannelOrder | nwp.Expr
    radius: _alt_t.ChannelRadius | nwp.Expr
    radius2: _alt_t.ChannelRadius2 | nwp.Expr
    row: _alt_t.ChannelRow | nwp.Expr
    shape: _alt_t.ChannelShape | nwp.Expr
    size: _alt_t.ChannelSize | nwp.Expr
    stroke: _alt_t.ChannelStroke | nwp.Expr
    strokeDash: _alt_t.ChannelStrokeDash | nwp.Expr
    strokeOpacity: _alt_t.ChannelStrokeOpacity | nwp.Expr
    strokeWidth: _alt_t.ChannelStrokeWidth | nwp.Expr
    text: _alt_t.ChannelText | nwp.Expr
    theta: _alt_t.ChannelTheta | nwp.Expr
    theta2: _alt_t.ChannelTheta2 | nwp.Expr
    time: str | alt.Time | alt.api.IntoCondition | nwp.Expr
    tooltip: _alt_t.ChannelTooltip | nwp.Expr
    url: _alt_t.ChannelUrl | nwp.Expr
    x: _alt_t.ChannelX | nwp.Expr
    x2: _alt_t.ChannelX2 | nwp.Expr
    xError: _alt_t.ChannelXError | nwp.Expr
    xError2: _alt_t.ChannelXError2 | nwp.Expr
    xOffset: _alt_t.ChannelXOffset | nwp.Expr
    y: _alt_t.ChannelY | nwp.Expr
    y2: _alt_t.ChannelY2 | nwp.Expr
    yError: _alt_t.ChannelYError | nwp.Expr
    yError2: _alt_t.ChannelYError2 | nwp.Expr
    yOffset: _alt_t.ChannelYOffset | nwp.Expr


if TYPE_CHECKING:
    # https://github.com/python/mypy/pull/21382
    class EncodeKwds(_EncodeKwds, TypedDict, closed=True): ...  # type: ignore[call-arg]
else:
    EncodeKwds = _EncodeKwds
