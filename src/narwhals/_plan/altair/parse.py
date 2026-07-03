from __future__ import annotations

from typing import TYPE_CHECKING, Final

from narwhals._plan import _parse, expressions as ir
from narwhals._plan.meta import resolve_name as _meta_resolve_name
from narwhals._plan.schema import FrozenSchema

if TYPE_CHECKING:
    from collections.abc import Iterator

    from narwhals._plan.altair.typing import IntoExprColumn, OutputName
    from narwhals._plan.typing import OneOrIterable


# required for the signature of `resolve_name`, but only for `struct`, which would fail here anyway
_EMPTY_SCHEMA: Final = FrozenSchema.empty()


def parse_into_named_exprs(
    exprs: OneOrIterable[IntoExprColumn] = (),
    *more_exprs: IntoExprColumn,
    **named_exprs: IntoExprColumn,
) -> Iterator[tuple[OutputName, ir.ExprIR]]:
    """Parse inputs for a select-like context."""
    for expr_ir in _parse.into_iter_expr_ir(exprs, *more_exprs, **named_exprs):
        # NOTE: I think altair's data model is too fuzzy for expression expansion
        # would be very nice to use selectors & `.name.{pre,suf}fix` if possible though
        yield _meta_resolve_name(expr_ir, _EMPTY_SCHEMA)
