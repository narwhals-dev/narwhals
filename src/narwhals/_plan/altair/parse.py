from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Final, TypeAlias

from narwhals._plan import _parse, expressions as ir
from narwhals._plan.meta import resolve_name as _meta_resolve_name
from narwhals._plan.schema import FrozenSchema
from narwhals._plan.typing import OneOrIterable

if TYPE_CHECKING:
    from collections.abc import Iterator

    import narwhals._plan as nwp
    from narwhals._plan.typing import OneOrIterable

    IntoExpr: TypeAlias = nwp.Expr | str
    """Only expressions or column names will ever be valid in this context."""

OutputName: TypeAlias = str


# required for the signature of `resolve_name`, but only for `struct`, which would fail here anyway
_EMPTY_SCHEMA: Final = FrozenSchema.empty()


def parse_into_named_exprs(
    exprs: OneOrIterable[IntoExpr] = (), *more_exprs: IntoExpr, **named_exprs: IntoExpr
) -> Iterator[tuple[OutputName, ir.ExprIR]]:
    """Use this for a select-like context."""
    for expr_ir in _parse.into_iter_expr_ir(exprs, *more_exprs, **named_exprs):
        # NOTE: I think altair's data model is too fuzzy for expression expansion
        # would be very nice to use selectors & `.name.{pre,suf}fix` if possible though
        yield _meta_resolve_name(expr_ir, _EMPTY_SCHEMA)
