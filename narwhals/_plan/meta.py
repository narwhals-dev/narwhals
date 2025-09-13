"""`pl.Expr.meta` namespace functionality.

- https://github.com/pola-rs/polars/blob/dafd0a2d0e32b52bcfa4273bffdd6071a0d5977a/crates/polars-plan/src/dsl/meta.rs#L11-L111
- https://github.com/pola-rs/polars/blob/dafd0a2d0e32b52bcfa4273bffdd6071a0d5977a/crates/polars-plan/src/plans/iterator.rs#L10-L105
"""

from __future__ import annotations

from functools import lru_cache
from itertools import chain
from typing import TYPE_CHECKING, Literal, overload

from narwhals._plan.common import IRNamespace
from narwhals.exceptions import ComputeError
from narwhals.utils import Version

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from typing_extensions import TypeIs

    from narwhals._plan.expressions import Column, ExprIR


class IRMetaNamespace(IRNamespace):
    """Methods to modify and traverse existing expressions."""

    def has_multiple_outputs(self) -> bool:
        return any(_has_multiple_outputs(e) for e in self._ir.iter_left())

    def is_column(self) -> bool:
        return is_column(self._ir)

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


def _expr_to_leaf_column_names_iter(ir: ExprIR) -> Iterator[str]:
    for e in _expr_to_leaf_column_exprs_iter(ir):
        result = _expr_to_leaf_column_name(e)
        if isinstance(result, str):
            yield result


def _expr_to_leaf_column_exprs_iter(ir: ExprIR) -> Iterator[ExprIR]:
    from narwhals._plan import expressions as expr

    for outer in ir.iter_root_names():
        if isinstance(outer, (expr.Column, expr.All)):
            yield outer


def _expr_to_leaf_column_name(ir: ExprIR) -> str | ComputeError:
    leaves = list(_expr_to_leaf_column_exprs_iter(ir))
    if not len(leaves) <= 1:
        msg = "found more than one root column name"
        return ComputeError(msg)
    if not leaves:
        msg = "no root column name found"
        return ComputeError(msg)
    leaf = leaves[0]
    from narwhals._plan import expressions as expr

    if isinstance(leaf, expr.Column):
        return leaf.name
    if isinstance(leaf, expr.All):
        msg = "wildcard has no root column name"
        return ComputeError(msg)
    msg = f"Expected unreachable, got {type(leaf).__name__!r}\n\n{leaf}"
    return ComputeError(msg)


def root_names_unique(irs: Iterable[ExprIR], /) -> set[str]:
    return set(chain.from_iterable(_expr_to_leaf_column_names_iter(e) for e in irs))


@lru_cache(maxsize=32)
def _expr_output_name(ir: ExprIR) -> str | ComputeError:
    from narwhals._plan import expressions as expr

    for e in ir.iter_output_name():
        if isinstance(e, (expr.Column, expr.Alias, expr.Literal, expr.Len)):
            return e.name
        if isinstance(e, (expr.All, expr.KeepName, expr.RenameAlias)):
            msg = "cannot determine output column without a context for this expression"
            return ComputeError(msg)
        if isinstance(e, (expr.Columns, expr.IndexColumns, expr.Nth)):
            msg = "this expression may produce multiple output names"
            return ComputeError(msg)
        continue
    msg = f"unable to find root column name for expr '{ir!r}' when calling 'output_name'"
    return ComputeError(msg)


def get_single_leaf_name(ir: ExprIR) -> str | ComputeError:
    """Find the name at the start of an expression.

    Normal iteration would just return the first root column it found.

    Based on [`polars_plan::utils::get_single_leaf`]

    [`polars_plan::utils::get_single_leaf`]: https://github.com/pola-rs/polars/blob/0fa7141ce718c6f0a4d6ae46865c867b177a59ed/crates/polars-plan/src/utils.rs#L151-L168
    """
    from narwhals._plan import expressions as expr

    for e in ir.iter_right():
        if isinstance(e, (expr.WindowExpr, expr.SortBy, expr.Filter)):
            return get_single_leaf_name(e.expr)
        if isinstance(e, expr.BinaryExpr):
            return get_single_leaf_name(e.left)
        # NOTE: `polars` doesn't include `Literal` here
        if isinstance(e, (expr.Column, expr.Len)):
            return e.name
    msg = f"unable to find a single leaf column in expr '{ir!r}'"
    return ComputeError(msg)


def _has_multiple_outputs(ir: ExprIR) -> bool:
    from narwhals._plan import expressions as expr

    return isinstance(ir, (expr.Columns, expr.IndexColumns, expr.SelectorIR, expr.All))


def has_expr_ir(ir: ExprIR, *matches: type[ExprIR]) -> bool:
    """Return True if any node in the tree is in type `matches`.

    Based on [`polars_plan::utils::has_expr`]

    [`polars_plan::utils::has_expr`]: https://github.com/pola-rs/polars/blob/0fa7141ce718c6f0a4d6ae46865c867b177a59ed/crates/polars-plan/src/utils.rs#L70-L77
    """
    return any(isinstance(e, matches) for e in ir.iter_right())


def is_column(ir: ExprIR) -> TypeIs[Column]:
    from narwhals._plan.expressions.expr import Column

    return isinstance(ir, Column)


def _is_literal(ir: ExprIR, *, allow_aliasing: bool) -> bool:
    from narwhals._plan import expressions as expr
    from narwhals._plan.expressions.literal import is_literal_scalar

    return (
        isinstance(ir, expr.Literal)
        or (allow_aliasing and isinstance(ir, expr.Alias))
        or (
            isinstance(ir, expr.Cast)
            and is_literal_scalar(ir.expr)
            and isinstance(ir.expr.dtype, Version.MAIN.dtypes.Datetime)
        )
    )


def _is_column_selection(ir: ExprIR, *, allow_aliasing: bool) -> bool:
    from narwhals._plan import expressions as expr

    return isinstance(ir, (expr.Column, expr._ColumnSelection, expr.SelectorIR)) or (
        allow_aliasing and isinstance(ir, (expr.Alias, expr.KeepName, expr.RenameAlias))
    )
