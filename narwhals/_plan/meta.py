"""`pl.Expr.meta` namespace functionality.

- https://github.com/pola-rs/polars/blob/dafd0a2d0e32b52bcfa4273bffdd6071a0d5977a/crates/polars-plan/src/dsl/meta.rs#L11-L111
- https://github.com/pola-rs/polars/blob/dafd0a2d0e32b52bcfa4273bffdd6071a0d5977a/crates/polars-plan/src/plans/iterator.rs#L10-L105
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Literal, overload

from narwhals._plan import expressions as ir
from narwhals._plan.expressions import selectors as cs
from narwhals._plan.expressions.literal import Lit, LitSeries
from narwhals._plan.expressions.namespace import IRNamespace
from narwhals._plan.expressions.struct import FieldByName
from narwhals._utils import unstable
from narwhals.exceptions import ComputeError
from narwhals.utils import Version

if TYPE_CHECKING:
    from collections.abc import Iterator

    from narwhals._plan import Selector


class MetaNamespace(IRNamespace):
    """Methods to traverse and introspect existing expressions."""

    def has_multiple_outputs(self) -> bool:
        """Indicate if this expression expands into multiple expressions."""
        return any(isinstance(e, ir.SelectorIR) for e in self._ir.iter_left())

    def is_column(self) -> bool:
        """Indicate if this expression is a basic/unaliased column."""
        return isinstance(self._ir, ir.Column)

    def is_column_selection(self, *, allow_aliasing: bool = False) -> bool:
        """Indicate if this expression only selects columns (optionally aliased).

        Arguments:
            allow_aliasing: If False (default), any aliasing is not considered to be column selection.
        """
        nodes = self._ir.iter_left()
        selection = ir.Column, ir.SelectorIR
        if not allow_aliasing:
            return all(isinstance(e, selection) for e in nodes)
        targets = *selection, ir.Alias, ir.KeepName, ir.RenameAlias
        return all(isinstance(e, targets) for e in nodes)

    def is_literal(self, *, allow_aliasing: bool = False) -> bool:
        """Indicate if this expression is a literal value (optionally aliased).

        Arguments:
            allow_aliasing: If False (default), only a bare literal will match.
        """
        it = self._ir.iter_left()
        selection = (Lit, LitSeries) if not allow_aliasing else (Lit, LitSeries, ir.Alias)
        return all(
            isinstance(e, selection)
            or (
                isinstance(e, ir.Cast)
                and isinstance(e.expr, Lit)
                and isinstance(e.expr.dtype, _TP_DATETIME)
            )
            for e in it
        )

    @overload
    def output_name(self, *, raise_if_undetermined: Literal[True] = True) -> str: ...
    @overload
    def output_name(self, *, raise_if_undetermined: Literal[False]) -> str | None: ...
    def output_name(self, *, raise_if_undetermined: bool = True) -> str | None:
        """Get the column name that this expression would produce.

        Arguments:
            raise_if_undetermined: If True (default), a `ComputeError` will be raised
                if the output name depends on the schema of the context.
                Otherwise `None` is returned.

        Examples:
            >>> import narwhals._plan as nw
            >>> import narwhals._plan.selectors as ncs
            >>> nw.col("a").meta.output_name()
            'a'

            Aliasing is supported:
            >>> nw.col("a").alias("b").meta.output_name()
            'b'

            And chained renaming operations:
            >>> nw.col("a").alias("b").min().name.to_uppercase().meta.output_name()
            'B'

            But selectors always require a schema:
            >>> ncs.string().meta.output_name(raise_if_undetermined=False)
            >>> ncs.string().meta.output_name()
            Traceback (most recent call last):
            narwhals.exceptions.ComputeError: unable to find root column name for expr 'ncs.string()' when calling 'output_name'
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
        return list(iter_root_names(self._ir))

    @unstable
    def as_selector(self) -> Selector:
        """Try to turn this expression into a selector.

        Raises if the underlying expression is not a column or selector.
        """
        return self._ir.to_selector_ir().to_narwhals()


_TP_DATETIME = Version.MAIN.dtypes.Datetime


def iter_root_names(expr: ir.ExprIR, /) -> Iterator[str]:
    yield from (e.name for e in expr.iter_left() if isinstance(e, ir.Column))


@lru_cache(maxsize=32)
def _expr_output_name(expr: ir.ExprIR, /) -> str | ComputeError:
    for e in expr.iter_output_name():
        if isinstance(e, (ir.Column, ir.Alias, ir.Lit, ir.LitSeries, ir.Len)):
            return e.name
        if isinstance(e, ir.RenameAlias):
            parent = _expr_output_name(e.expr)
            return e.function(parent) if isinstance(parent, str) else parent
        if isinstance(e, ir.KeepName):
            msg = "cannot determine output column without a context for this expression"
            return ComputeError(msg)
        if isinstance(e, cs.ByName) and len(e.names) == 1:
            return e.names[0]
        if isinstance(e, ir.StructExpr) and isinstance(e.function, FieldByName):
            return e.function.name
    msg = (
        f"unable to find root column name for expr '{expr!r}' when calling 'output_name'"
    )
    return ComputeError(msg)
