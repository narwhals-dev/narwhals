from __future__ import annotations

from importlib.util import find_spec

from narwhals.typing import PythonLiteral

if find_spec("altair") is None:
    msg = "`altair` is required to convert `ExprIR`s"
    raise ModuleNotFoundError(msg)

from typing import TYPE_CHECKING, TypeAlias, Union

from altair.vegalite.v6.schema import _typing as alt_t

if TYPE_CHECKING:
    import narwhals._plan as nwp


VegaExpr: TypeAlias = str
"""A stringized [Vega Expression].

[Vega Expression]: https://vega.github.io/vega/docs/expressions/
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


AggregateOp: TypeAlias = alt_t.AggregateOp_T
"""Aggregates accepted by `WindowFieldDef(op=...)`."""

WindowOp: TypeAlias = alt_t.WindowOnlyOp_T
"""Window functions accepted by `WindowFieldDef(op=...)`."""

AggOrWindow: TypeAlias = AggregateOp | WindowOp
"""Anything accepted by `WindowFieldDef(op=...)`."""
