"""Based on [polars-plan/src/plans/conversion/expr_expansion.rs].

- Goal is to expand every selection into a named column.
- Most will require only the column names of the schema.

[polars-plan/src/plans/conversion/expr_expansion.rs]: https://github.com/pola-rs/polars/blob/df4d21c30c2b383b651e194f8263244f2afaeda3/crates/polars-plan/src/plans/conversion/expr_expansion.rs
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._plan.common import Immutable

if TYPE_CHECKING:
    from narwhals._plan.common import ExprIR
    from narwhals._plan.dummy import DummyExpr


class ExpansionFlags(Immutable):
    """`polars` uses a struct, but we may want to use `enum.Flag`."""

    __slots__ = (
        "has_exclude",
        "has_nth",
        "has_selector",
        "has_wildcard",
        "multiple_columns",
    )
    multiple_columns: bool
    has_nth: bool
    has_wildcard: bool
    has_selector: bool
    has_exclude: bool

    @property
    def expands(self) -> bool:
        """If we add struct stuff, that would slot in here as well."""
        return self.multiple_columns

    @staticmethod
    def from_ir(ir: ExprIR, /) -> ExpansionFlags:
        """Subset of [`find_flags`].

        [`find_flags`]: https://github.com/pola-rs/polars/blob/df4d21c30c2b383b651e194f8263244f2afaeda3/crates/polars-plan/src/plans/conversion/expr_expansion.rs#L607-L660
        """
        from narwhals._plan import expr

        multiple_columns: bool = False
        has_nth: bool = False
        has_wildcard: bool = False
        has_selector: bool = False
        has_exclude: bool = False
        for e in ir.iter_left():
            if isinstance(e, (expr.Columns, expr.IndexColumns)):
                multiple_columns = True
            elif isinstance(e, expr.Nth):
                has_nth = True
            elif isinstance(e, expr.All):
                has_wildcard = True
            elif isinstance(e, expr.SelectorIR):
                has_selector = True
            elif isinstance(e, expr.Exclude):
                has_exclude = True
        return ExpansionFlags(
            multiple_columns=multiple_columns,
            has_nth=has_nth,
            has_wildcard=has_wildcard,
            has_selector=has_selector,
            has_exclude=has_exclude,
        )

    @classmethod
    def from_expr(cls, expr: DummyExpr, /) -> ExpansionFlags:
        return cls.from_ir(expr._ir)
