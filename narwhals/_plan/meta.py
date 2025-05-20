"""`pl.Expr.meta` namespace functionality.

- It seems like there might be a need to distinguish the top-level nodes for iterating
  - polars_plan::dsl::expr::Expr
- https://github.com/pola-rs/polars/blob/dafd0a2d0e32b52bcfa4273bffdd6071a0d5977a/crates/polars-plan/src/dsl/meta.rs#L11-L111
- https://github.com/pola-rs/polars/blob/dafd0a2d0e32b52bcfa4273bffdd6071a0d5977a/crates/polars-plan/src/plans/iterator.rs#L10-L105
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

    import polars as pl

    from narwhals._plan.common import ExprIR


class ExprIRMetaNamespace:
    """Requires defining iterator behavior per node."""

    def __init__(self, ir: ExprIR, /) -> None:
        self._ir: ExprIR = ir

    def has_multiple_outputs(self) -> bool:
        raise NotImplementedError

    def is_column(self) -> bool:
        """Only one that doesn't require iter.

        https://github.com/pola-rs/polars/blob/dafd0a2d0e32b52bcfa4273bffdd6071a0d5977a/crates/polars-plan/src/dsl/meta.rs#L65-L71
        """
        from narwhals._plan.expr import Column

        return isinstance(self._ir, Column)

    def is_column_selection(self, *, allow_aliasing: bool = False) -> bool:
        raise NotImplementedError

    def is_literal(self, *, allow_aliasing: bool = False) -> bool:
        raise NotImplementedError

    def output_name(self, *, raise_if_undetermined: bool = True) -> str | None:
        """https://github.com/pola-rs/polars/blob/dafd0a2d0e32b52bcfa4273bffdd6071a0d5977a/crates/polars-plan/src/utils.rs#L126-L127."""
        raise NotImplementedError

    # NOTE: Less important for us, but maybe nice to have
    def pop(self) -> list[ExprIR]:
        raise NotImplementedError

    def root_names(self) -> list[str]:
        raise NotImplementedError

    def undo_aliases(self) -> ExprIR:
        raise NotImplementedError

    # NOTE: We don't support `nw.col("*")` or other patterns in col
    # Maybe not relevant at all
    def is_regex_projection(self) -> bool:
        raise NotImplementedError


def profile_polars_expr(expr: pl.Expr) -> dict[str, Any]:
    """Gather all metadata for a native `Expr`.

    Eventual goal would be that a `nw.Expr` matches a `pl.Expr` in as much of this as possible.
    """
    return {
        "has_multiple_outputs": expr.meta.has_multiple_outputs(),
        "is_column": expr.meta.is_column(),
        "is_regex_projection": expr.meta.is_regex_projection(),
        "is_column_selection": expr.meta.is_column_selection(),
        "is_column_selection(allow_aliasing=True)": expr.meta.is_column_selection(
            allow_aliasing=True
        ),
        "is_literal": expr.meta.is_literal(),
        "is_literal(allow_aliasing=True)": expr.meta.is_literal(allow_aliasing=True),
        "output_name": expr.meta.output_name(raise_if_undetermined=False),
        "root_names": expr.meta.root_names(),
        "pop": expr.meta.pop(),
        "undo_aliases": expr.meta.undo_aliases(),
        "expr": expr,
    }
