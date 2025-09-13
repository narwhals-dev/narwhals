"""`pl.Expr.meta` namespace functionality.

- https://github.com/pola-rs/polars/blob/dafd0a2d0e32b52bcfa4273bffdd6071a0d5977a/crates/polars-plan/src/dsl/meta.rs#L11-L111
- https://github.com/pola-rs/polars/blob/dafd0a2d0e32b52bcfa4273bffdd6071a0d5977a/crates/polars-plan/src/plans/iterator.rs#L10-L105
"""

from __future__ import annotations

from functools import lru_cache
from itertools import chain
from typing import TYPE_CHECKING, Literal, overload

from narwhals._plan import expressions as ir
from narwhals._plan._guards import is_literal
from narwhals._plan.expressions.literal import is_literal_scalar
from narwhals._plan.expressions.namespace import IRNamespace
from narwhals.exceptions import ComputeError
from narwhals.utils import Version

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


class MetaNamespace(IRNamespace):
    """Methods to modify and traverse existing expressions."""

    def has_multiple_outputs(self) -> bool:
        return any(_has_multiple_outputs(e) for e in self._ir.iter_left())

    def is_column(self) -> bool:
        return isinstance(self._ir, ir.Column)

    def is_column_selection(self, *, allow_aliasing: bool = False) -> bool:
        return all(
            _is_column_selection(e, allow_aliasing=allow_aliasing)
            for e in self._ir.iter_left()
        )

    def is_literal(self, *, allow_aliasing: bool = False) -> bool:
        return all(
            _is_literal(e, allow_aliasing=allow_aliasing) for e in self._ir.iter_left()
        )

    @overload
    def output_name(self, *, raise_if_undetermined: Literal[True] = True) -> str: ...
    @overload
    def output_name(self, *, raise_if_undetermined: Literal[False]) -> str | None: ...
    def output_name(self, *, raise_if_undetermined: bool = True) -> str | None:
        """Get the output name of this expression.

        Examples:
            >>> from narwhals._plan import functions as nwd
            >>>
            >>> a = nwd.col("a")
            >>> b = a.alias("b")
            >>> c = b.min().alias("c")
            >>>
            >>> a.meta.output_name()
            'a'
            >>> b.meta.output_name()
            'b'
            >>> c.meta.output_name()
            'c'
        """
        ok_or_err = _expr_output_name(self._ir)
        if isinstance(ok_or_err, ComputeError):
            if raise_if_undetermined:
                # NOTE: See (https://github.com/narwhals-dev/narwhals/pull/2572#discussion_r2161824883)
                _expr_output_name.cache_clear()
                raise ok_or_err
            return None
        return ok_or_err

    def root_names(self) -> list[str]:
        """Get the root column names."""
        return list(_expr_to_leaf_column_names_iter(self._ir))


def _expr_to_leaf_column_names_iter(expr: ir.ExprIR, /) -> Iterator[str]:
    for e in _expr_to_leaf_column_exprs_iter(expr):
        result = _expr_to_leaf_column_name(e)
        if isinstance(result, str):
            yield result


def _expr_to_leaf_column_exprs_iter(expr: ir.ExprIR, /) -> Iterator[ir.ExprIR]:
    for outer in expr.iter_root_names():
        if isinstance(outer, (ir.Column, ir.All)):
            yield outer


def _expr_to_leaf_column_name(expr: ir.ExprIR, /) -> str | ComputeError:
    leaves = list(_expr_to_leaf_column_exprs_iter(expr))
    if not len(leaves) <= 1:
        msg = "found more than one root column name"
        return ComputeError(msg)
    if not leaves:
        msg = "no root column name found"
        return ComputeError(msg)
    leaf = leaves[0]
    if isinstance(leaf, ir.Column):
        return leaf.name
    if isinstance(leaf, ir.All):
        msg = "wildcard has no root column name"
        return ComputeError(msg)
    msg = f"Expected unreachable, got {type(leaf).__name__!r}\n\n{leaf}"
    return ComputeError(msg)


def root_names_unique(exprs: Iterable[ir.ExprIR], /) -> set[str]:
    return set(chain.from_iterable(_expr_to_leaf_column_names_iter(e) for e in exprs))


@lru_cache(maxsize=32)
def _expr_output_name(expr: ir.ExprIR, /) -> str | ComputeError:
    for e in expr.iter_output_name():
        if isinstance(e, (ir.Column, ir.Alias, ir.Literal, ir.Len)):
            return e.name
        if isinstance(e, (ir.All, ir.KeepName, ir.RenameAlias)):
            msg = "cannot determine output column without a context for this expression"
            return ComputeError(msg)
        if isinstance(e, (ir.Columns, ir.IndexColumns, ir.Nth)):
            msg = "this expression may produce multiple output names"
            return ComputeError(msg)
        continue
    msg = (
        f"unable to find root column name for expr '{expr!r}' when calling 'output_name'"
    )
    return ComputeError(msg)


def get_single_leaf_name(expr: ir.ExprIR, /) -> str | ComputeError:
    """Find the name at the start of an expression.

    Normal iteration would just return the first root column it found.

    Based on [`polars_plan::utils::get_single_leaf`]

    [`polars_plan::utils::get_single_leaf`]: https://github.com/pola-rs/polars/blob/0fa7141ce718c6f0a4d6ae46865c867b177a59ed/crates/polars-plan/src/utils.rs#L151-L168
    """
    for e in expr.iter_right():
        if isinstance(e, (ir.WindowExpr, ir.SortBy, ir.Filter)):
            return get_single_leaf_name(e.expr)
        if isinstance(e, ir.BinaryExpr):
            return get_single_leaf_name(e.left)
        # NOTE: `polars` doesn't include `Literal` here
        if isinstance(e, (ir.Column, ir.Len)):
            return e.name
    msg = f"unable to find a single leaf column in expr '{expr!r}'"
    return ComputeError(msg)


def _has_multiple_outputs(expr: ir.ExprIR, /) -> bool:
    return isinstance(expr, (ir.Columns, ir.IndexColumns, ir.SelectorIR, ir.All))


def has_expr_ir(expr: ir.ExprIR, *matches: type[ir.ExprIR]) -> bool:
    """Return True if any node in the tree is in type `matches`.

    Based on [`polars_plan::utils::has_expr`]

    [`polars_plan::utils::has_expr`]: https://github.com/pola-rs/polars/blob/0fa7141ce718c6f0a4d6ae46865c867b177a59ed/crates/polars-plan/src/utils.rs#L70-L77
    """
    return any(isinstance(e, matches) for e in expr.iter_right())


def _is_literal(expr: ir.ExprIR, /, *, allow_aliasing: bool) -> bool:
    return (
        is_literal(expr)
        or (allow_aliasing and isinstance(expr, ir.Alias))
        or (
            isinstance(expr, ir.Cast)
            and is_literal_scalar(expr.expr)
            and isinstance(expr.expr.dtype, Version.MAIN.dtypes.Datetime)
        )
    )


def _is_column_selection(expr: ir.ExprIR, /, *, allow_aliasing: bool) -> bool:
    return isinstance(expr, (ir.Column, ir._ColumnSelection, ir.SelectorIR)) or (
        allow_aliasing and isinstance(expr, (ir.Alias, ir.KeepName, ir.RenameAlias))
    )
