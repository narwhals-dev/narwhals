# TODO @dangotbanned: (medium-priority) Module doc is a bit dated
# - Does this still reflect current narwhals?
# - Is this content useful to see here?
"""Expanding expressions/selectors.

Based on [polars-plan/src/plans/conversion/expr_expansion.rs].

## Notes
- Goal is to expand every selection into a named column.
- Most will require only the column names of the schema.

## Current `narwhals`
As of [6e57eff4f059c748cf84ddcae276a74318720b85], many of the problems
this module would solve *currently* have solutions distributed throughout `narwhals`.

Their dependencies are **quite** complex, with the main ones being:
- `CompliantExpr`
  - _evaluate_output_names
    - `CompliantSelector.__(sub|or|and|invert)__`
    - `CompliantThen._evaluate_output_names`
  - _alias_output_names
  - from_column_names, from_column_indices
    - `CompliantNamespace.(all|col|exclude|nth)`
  - _eval_names_indices
  - _evaluate_aliases
    - `Compliant*Frame._evaluate_aliases`
    - `EagerDataFrame._evaluate_into_expr(s)`
- `CompliantExprNameNamespace`
  - EagerExprNameNamespace
  - LazyExprNameNamespace
- `_expression_parsing.py`
  - combine_evaluate_output_names
    - 6-7x per `CompliantNamespace`
  - combine_alias_output_names
    - 6-7x per `CompliantNamespace`
  - evaluate_output_names_and_aliases
    - Depth tracking (`Expr.over`, `GroupyBy.agg`)

[polars-plan/src/plans/conversion/expr_expansion.rs]: https://github.com/pola-rs/polars/blob/df4d21c30c2b383b651e194f8263244f2afaeda3/crates/polars-plan/src/plans/conversion/expr_expansion.rs
[6e57eff4f059c748cf84ddcae276a74318720b85]: https://github.com/narwhals-dev/narwhals/commit/6e57eff4f059c748cf84ddcae276a74318720b85
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from itertools import chain
from typing import TYPE_CHECKING

from narwhals._plan import expressions as ir, meta
from narwhals._plan._parse import into_iter_selector_ir
from narwhals._plan.exceptions import (
    column_not_found_error,
    duplicate_error,
    duplicate_names_error,
    expand_multi_output_error,
    selectors_not_found_error,
)
from narwhals._plan.expressions import (
    Alias,
    ExprIR,
    KeepName,
    NamedIR,
    RenameAlias,
    SelectorIR,
)
from narwhals._plan.schema import FrozenSchema, IntoFrozenSchema
from narwhals.exceptions import InvalidOperationError

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable

    from narwhals._plan.selectors import Selector
    from narwhals._plan.typing import Ignored, OneOrIterable, OutputNames, Seq

__all__ = ["Expander", "expand_selectors", "parse_expand_selectors", "prepare_projection"]


def prepare_projection(
    exprs: Collection[ExprIR], /, ignored: Ignored = (), *, schema: IntoFrozenSchema
) -> tuple[Seq[NamedIR], FrozenSchema]:
    """Expand expressions into named column projections.

    Entry-point for a [projection context], which will execute the resolved expressions.

    Arguments:
        exprs: Expressions to project.
        ignored: Names of `group_by` key columns.
            Required for projecting aggregations in `group_by`.
        schema: Scope to expand selectors, validate selections and resolve renaming operations.

    [projection context]: https://docs.pola.rs/user-guide/concepts/expressions-and-contexts/#contexts
    """
    expander = Expander(schema, ignored)
    projected, _ = expander.prepare_projection(exprs)
    return projected, expander.schema


def expand_selectors(
    selectors: Iterable[SelectorIR],
    /,
    *,
    schema: IntoFrozenSchema,
    require_any: bool = True,
) -> OutputNames:
    """Expand selectors into the column names that match.

    Provides selector-support (widely) across frame-level APIs, where the full scope of
    `prepare_projection` is not required.

    Arguments:
        selectors: Exclusively selector-only input.
        schema: Scope to expand selectors in.
        require_any: If True (default) raise if the entire expansion selected zero columns.
            If False, we can always defer iterator collection until finishing expansion.
    """
    return Expander(schema).expand_selectors(selectors, require_any=require_any)


def parse_expand_selectors(
    first_input: OneOrIterable[str | Selector],
    more_inputs: tuple[str | Selector, ...] = (),
    /,
    *,
    schema: IntoFrozenSchema,
    require_any: bool = True,
) -> OutputNames:
    """Convert input(s) into selector(s), expanding them into the column names that match.

    Equivalent to these independent steps, ensuring things stay lazy as long as possible:

        irs: Iterator[SelectorIR] = _parse.into_iter_selector_ir(first_input, more_inputs)
        output_names: tuple[str, ...] = expand_selectors(irs, schema=..., require_any=...)

    Arguments:
        first_input: One or more column names or selectors.
        more_inputs: Use if `*args` were accepted *in-addition-to* `first_input` as syntax sugar.
        schema: Scope to expand selectors in.
        require_any: If True (default) raise if the entire expansion selected zero columns.
            If False, we can always defer iterator collection until finishing expansion.
    """
    parsed = into_iter_selector_ir(first_input, more_inputs)
    return expand_selectors(parsed, schema=schema, require_any=require_any)


def _is_duplicated(names: Collection[str]) -> bool:
    return len(names) != len(set(names))


def _remove_alias(origin: ExprIR, /) -> ExprIR:
    def fn(child: ExprIR, /) -> ExprIR:
        return child.expr if isinstance(child, (Alias, RenameAlias)) else child

    return origin.map_ir(fn)


def _replace_keep_name(origin: ExprIR, /) -> ExprIR:
    if (name := next(meta.iter_root_names(origin), None)) is None:
        msg = f"`name.keep` expected at least one column name, got `{origin!r}`"
        raise InvalidOperationError(msg)

    def fn(child: ExprIR, /) -> ExprIR:
        return child.expr.alias(name) if isinstance(child, KeepName) else child

    return origin.map_ir(fn)


class Expander:
    """Expand multiple expressions against a target schema.

    Provides a context for resolving and validating these transformations:

        Iterable[SelectorIR] -> tuple[str, ...]
        Collection[ExprIR]   -> tuple[NamedIR[ExprIR], ...]

    Arguments:
        schema: Target scope for expansion/validation.
        ignored: Names of `group_by` key columns.

    Important:
        Adapted from [upstream].

    [upstream]: https://github.com/pola-rs/polars/blob/3291151b5a0e6fa82658cbad5f9b9c6aec3905a6/crates/polars-plan/src/plans/conversion/dsl_to_ir/expr_expansion.rs
    """

    __slots__ = ("ignored", "schema")
    schema: FrozenSchema
    ignored: Ignored

    def __init__(self, schema: IntoFrozenSchema, ignored: Ignored = ()) -> None:
        self.schema = FrozenSchema(schema)
        self.ignored = ignored

    def _iter_expand_expressions(self, exprs: Iterable[ExprIR], /) -> Iterator[ExprIR]:
        for expr in exprs:
            if any(e.needs_expansion() for e in expr.iter_left()):
                yield from expr.iter_expand(self)
            else:
                yield expr

    def _iter_expand_selectors(self, selectors: Iterable[SelectorIR], /) -> Iterator[str]:
        for s in selectors:
            yield from s.iter_expand_selector(self.schema)

    def expand_selectors(
        self,
        selectors: Iterable[SelectorIR],
        /,
        *,
        check_unique: bool = True,
        require_any: bool = True,
    ) -> OutputNames:
        """Expand selectors into the column names that match."""
        if require_any and isinstance(selectors, Iterator):
            # Ensure we can show the original selectors in `ColumnNotFoundError`
            selectors = tuple(selectors)
        names = tuple(self._iter_expand_selectors(selectors))
        if require_any and not names:
            raise selectors_not_found_error(selectors, self.schema)
        if check_unique and _is_duplicated(names):
            raise duplicate_names_error(names)
        return names

    def prepare_projection(
        self, exprs: Collection[ExprIR], /
    ) -> tuple[Seq[NamedIR], deque[str]]:
        """Expand expressions into named column projections.

        Provides full validation (duplicates, missing columns) and resolving renaming operations.
        """
        # NOTE: Returning a deque was a hack to expose `output_names`,
        # without changing the signature of `_expansion.prepare_projection`
        # https://github.com/narwhals-dev/narwhals/commit/cef6c4673b2955d311ee5ecc091777b84ba9b73e
        output_names = deque[str]()
        named_irs = deque[NamedIR]()
        root_names = deque[Iterator[str]]()
        expand = self._iter_expand_expressions
        for e in expand(exprs):
            # NOTE: "" is allowed as a name, but falsy
            if (name := e.meta.output_name(raise_if_undetermined=False)) is not None:
                target = _remove_alias(e)
            else:
                replaced = _replace_keep_name(e)
                name = replaced.meta.output_name()
                target = _remove_alias(replaced)
            output_names.append(name)
            named_irs.append(ir.NamedIR(name, target))
            root_names.append(meta.iter_root_names(e))

        # NOTE: On failure, we repeat the expansion so the happy path doesn't need to collect as much
        if _is_duplicated(output_names):
            raise duplicate_error(tuple(expand(exprs)))
        if not self.schema.contains_all(root_names):
            roots = chain.from_iterable(meta.iter_root_names(e) for e in expand(exprs))
            raise column_not_found_error(roots, self.schema)
        return tuple(named_irs), output_names

    # TODO @dangotbanned: Does it still make sense for this to live here?
    def only(self, origin: ExprIR, child: ExprIR, /) -> ExprIR:
        """Expand a node, ensuring it produces a single output.

        Allows us to support any selector in more places, with the restriction being schema-dependent.
        """
        iterable = child.iter_expand(self)
        first = next(iterable)
        if second := next(iterable, None):
            raise expand_multi_output_error(origin, child, first, second, *iterable)
        return first
