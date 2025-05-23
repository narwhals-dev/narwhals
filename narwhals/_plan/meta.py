"""`pl.Expr.meta` namespace functionality.

- It seems like there might be a need to distinguish the top-level nodes for iterating
  - polars_plan::dsl::expr::Expr
- https://github.com/pola-rs/polars/blob/dafd0a2d0e32b52bcfa4273bffdd6071a0d5977a/crates/polars-plan/src/dsl/meta.rs#L11-L111
- https://github.com/pola-rs/polars/blob/dafd0a2d0e32b52bcfa4273bffdd6071a0d5977a/crates/polars-plan/src/plans/iterator.rs#L10-L105
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._plan.common import ExprIRNamespace
from narwhals.exceptions import ComputeError
from narwhals.utils import Version

if TYPE_CHECKING:
    from typing import Any

    import polars as pl

    from narwhals._plan.common import ExprIR


class ExprIRMetaNamespace(ExprIRNamespace):
    """Methods to modify and traverse existing expressions."""

    def has_multiple_outputs(self) -> bool:
        return any(_has_multiple_outputs(e) for e in self.ir.iter_left())

    def is_column(self) -> bool:
        from narwhals._plan.expr import Column

        return isinstance(self.ir, Column)

    def is_column_selection(self, *, allow_aliasing: bool = False) -> bool:
        return all(
            _is_column_selection(e, allow_aliasing=allow_aliasing)
            for e in self.ir.iter_left()
        )

    def is_literal(self, *, allow_aliasing: bool = False) -> bool:
        return all(
            _is_literal(e, allow_aliasing=allow_aliasing) for e in self.ir.iter_left()
        )

    def output_name(self, *, raise_if_undetermined: bool = True) -> str | None:
        """Get the output name of this expression.

        Examples:
            >>> from narwhals._plan import demo as nwd
            >>>
            >>> a = nwd.col("a")
            >>> b = a.alias("b")
            >>> c = b.min().alias("c")
            >>> c_over = c.over(nwd.col("e"), nwd.col("f"))
            >>> c_over_sort = c_over.sort_by(nwd.nth(9), nwd.col("g", "h"))
            >>>
            >>> a.meta.output_name()
            'a'
            >>> b.meta.output_name()
            'b'
            >>> c.meta.output_name()
            'c'
            >>> c_over.meta.output_name()
            'c'
            >>> c_over_sort.meta.output_name()
            'c'
            >>> nwd.lit(1).meta.output_name()
            'literal'
            >>> nwd.len().meta.output_name()
            'len'
        """
        ok_or_err = _expr_output_name(self.ir)
        if isinstance(ok_or_err, ComputeError):
            if raise_if_undetermined:
                raise ok_or_err
            return None
        return ok_or_err

    # NOTE: Less important for us, but maybe nice to have
    def pop(self) -> list[ExprIR]:
        raise NotImplementedError

    def root_names(self) -> list[str]:
        raise NotImplementedError

    def undo_aliases(self) -> ExprIR:
        raise NotImplementedError


def _expr_output_name(ir: ExprIR) -> str | ComputeError:
    from narwhals._plan import expr

    for e in ir.iter_right():
        if isinstance(e, (expr.WindowExpr, expr.SortBy)):
            # Don't follow `over(partition_by=...)` or `sort_by(by=...)
            return _expr_output_name(e.expr)
        if isinstance(e, (expr.Column, expr.Alias, expr.Literal, expr.Len)):
            return e.name
        if isinstance(e, expr.All):
            msg = "cannot determine output column without a context for this expression"
            return ComputeError(msg)
        if isinstance(e, (expr.Columns, expr.IndexColumns, expr.Nth)):
            msg = "this expression may produce multiple output names"
            return ComputeError(msg)
        continue
    msg = f"unable to find root column name for expr '{ir!r}' when calling 'output_name'"
    return ComputeError(msg)


def _has_multiple_outputs(ir: ExprIR) -> bool:
    from narwhals._plan import expr

    return isinstance(ir, (expr.Columns, expr.IndexColumns, expr.SelectorIR, expr.All))


def _is_literal(ir: ExprIR, *, allow_aliasing: bool) -> bool:
    from narwhals._plan import expr
    from narwhals._plan.literal import ScalarLiteral

    if isinstance(ir, expr.Literal):
        return True
    if isinstance(ir, expr.Alias):
        return allow_aliasing
    if isinstance(ir, expr.Cast):
        return (
            isinstance(ir.expr, expr.Literal)
            and isinstance(ir.expr, ScalarLiteral)
            and isinstance(ir.expr.dtype, Version.MAIN.dtypes.Datetime)
        )
    return False


def _is_column_selection(ir: ExprIR, *, allow_aliasing: bool) -> bool:
    from narwhals._plan import expr

    if isinstance(
        ir,
        (
            expr.Column,
            expr.Columns,
            expr.Exclude,
            expr.Nth,
            expr.IndexColumns,
            expr.SelectorIR,
            expr.All,
        ),
    ):
        return True
    # TODO @dangotbanned: Add `KeepName`, `RenameAlias` here later (see `_plan.name`)
    aliasing_types = (expr.Alias,)
    if isinstance(ir, aliasing_types):
        return allow_aliasing
    return False


def polars_expr_metadata(expr: pl.Expr) -> dict[str, Any]:
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


def polars_expr_to_dict(expr: pl.Expr) -> dict[str, Any]:
    """Serialize a native `Expr`, roundtrip back to `dict`.

    Using to inspect [`FunctionOptions`] and ensure we combine them in a similar way.

    [`FunctionOptions`]: https://github.com/narwhals-dev/narwhals/pull/2572#issuecomment-2891577685
    """
    import json

    return json.loads(expr.meta.serialize(format="json"))  # type: ignore[no-any-return]
