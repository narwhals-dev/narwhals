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
    from collections.abc import Collection, Iterable, Sequence

    from typing_extensions import TypeAlias

    from narwhals._plan.selectors import Selector
    from narwhals._plan.typing import Ignored, OneOrIterable, Seq

OutputNames: TypeAlias = "Seq[str]"
"""Fully expanded, validated output column names, for `NamedIR`s."""


def prepare_projection(
    exprs: Sequence[ExprIR], /, ignored: Ignored = (), *, schema: IntoFrozenSchema
) -> tuple[Seq[NamedIR], FrozenSchema]:
    """Expand IRs into named column projections.

    **Primary entry-point**, for `select`, `with_columns`,
    and any other context that requires resolving expression names.

    Arguments:
        exprs: IRs that *may* contain arbitrarily nested expressions.
        ignored: Names of `group_by` columns.
        schema: Scope to expand selectors in.
    """
    expander = Expander(schema, ignored)
    return expander.prepare_projection(exprs), expander.schema


def expand_selectors(
    selectors: Iterable[SelectorIR],
    /,
    *,
    schema: IntoFrozenSchema,
    require_any: bool = True,
) -> OutputNames:
    """Expand selector-only input into the column names that match.

    Similar to `prepare_projection`, but intended for allowing a subset of `Expr` and all `Selector`s
    to be used in more places like `DataFrame.{drop,sort,partition_by}`.

    Arguments:
        selectors: IRs that **only** contain subclasses of `SelectorIR`.
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

    Semantically equivalent to these independent steps:

        irs: Iterator[SelectorIR] = _parse.into_iter_selector_ir(first_input, more_inputs)
        output_names: tuple[str, ...] = expand_selectors(irs, schema=..., require_any=...)

    With the possibility of performing the entire operation in a single pass.

    Arguments:
        first_input: One or more column names or selectors.
        more_inputs: Use if `*args` were accepted *in-addition-to* `first_input` as syntax sugar.
        schema: Scope to expand selectors in.
        require_any: If True (default) raise if the entire expansion selected zero columns.
            If False, we can always defer iterator collection until finishing expansion.
    """
    parsed = into_iter_selector_ir(first_input, more_inputs)
    return expand_selectors(parsed, schema=schema, require_any=require_any)


def is_duplicated(names: Collection[str]) -> bool:
    return len(names) != len(set(names))


def remove_alias(origin: ExprIR, /) -> ExprIR:
    def fn(child: ExprIR, /) -> ExprIR:
        return child.expr if isinstance(child, (Alias, RenameAlias)) else child

    return origin.map_ir(fn)


def replace_keep_name(origin: ExprIR, /) -> ExprIR:
    if (name := next(meta.iter_root_names(origin), None)) is None:
        msg = f"`name.keep` expected at least one column name, got `{origin!r}`"
        raise InvalidOperationError(msg)

    def fn(child: ExprIR, /) -> ExprIR:
        return child.expr.alias(name) if isinstance(child, KeepName) else child

    return origin.map_ir(fn)


class Expander:
    # Based on https://github.com/pola-rs/polars/blob/5b90db75911c70010d0c0a6941046e6144af88d4/crates/polars-plan/src/plans/conversion/dsl_to_ir/expr_expansion.rs#L253-L850
    __slots__ = ("ignored", "schema")
    schema: FrozenSchema
    ignored: Ignored

    def __init__(self, scope: IntoFrozenSchema, ignored: Ignored = ()) -> None:
        self.schema = FrozenSchema(scope)
        self.ignored = ignored

    def iter_expand_expressions(self, exprs: Iterable[ExprIR], /) -> Iterator[ExprIR]:
        """Expand multiple expressions within this context.

        Matches selectors, converts them to columns and yields the transformed results.

        Use `prepare_projection` for full validation (duplicates, missing columns) and
        resolving renaming operations.
        """
        for expr in exprs:
            if any(e.needs_expansion() for e in expr.iter_left()):
                yield from expr.iter_expand(self)
            else:
                yield expr

    def iter_expand_selectors(self, selectors: Iterable[SelectorIR], /) -> Iterator[str]:
        for s in selectors:
            yield from s.iter_expand_selector(self.schema, self.ignored)

    def expand_selectors(
        self,
        selectors: Iterable[SelectorIR],
        /,
        *,
        check_unique: bool = True,
        require_any: bool = True,
    ) -> OutputNames:
        if require_any and isinstance(selectors, Iterator):
            # Ensure we can show the original selectors in `ColumnNotFoundError`
            selectors = tuple(selectors)
        names = tuple(self.iter_expand_selectors(selectors))
        if require_any and not names:
            raise selectors_not_found_error(selectors, self.schema)
        if check_unique and is_duplicated(names):
            raise duplicate_names_error(names)
        return names

    def prepare_projection(self, exprs: Collection[ExprIR], /) -> Seq[NamedIR]:
        named_irs, _ = self._prepare_projection(exprs)
        return named_irs

    # NOTE: Making this private was a hack to expose the collected `output_names`,
    # without changing the signature of `prepare_projection`
    # https://github.com/narwhals-dev/narwhals/commit/cef6c4673b2955d311ee5ecc091777b84ba9b73e
    def _prepare_projection(
        self, exprs: Collection[ExprIR], /
    ) -> tuple[Seq[NamedIR], deque[str]]:
        output_names = deque[str]()
        named_irs = deque[NamedIR]()
        root_names = deque[Iterator[str]]()
        expand = self.iter_expand_expressions
        for e in expand(exprs):
            # NOTE: "" is allowed as a name, but falsy
            if (name := e.meta.output_name(raise_if_undetermined=False)) is not None:
                target = remove_alias(e)
            else:
                replaced = replace_keep_name(e)
                name = replaced.meta.output_name()
                target = remove_alias(replaced)
            output_names.append(name)
            named_irs.append(ir.NamedIR(name, target))
            root_names.append(meta.iter_root_names(e))

        # NOTE: On failure, we repeat the expansion so the happy path doesn't need to collect as much
        if is_duplicated(output_names):
            raise duplicate_error(tuple(expand(exprs)))
        if not self.schema.contains_all(root_names):
            roots = chain.from_iterable(meta.iter_root_names(e) for e in expand(exprs))
            raise column_not_found_error(roots, self.schema)
        return tuple(named_irs), output_names

    # TODO @dangotbanned: Find a new home for some version of this doc
    def inner(self, children: Seq[ExprIR], /) -> Seq[ExprIR]:  # pragma: no cover
        """Expand a non-root node, *without* duplicating the root.

        If we wrote:

            col("a").over(col("c", "d", "e"))

        Then the expanded version should be:

            col("a").over(col("c"), col("d"), col("e"))

        An **incorrect** output would cause an error without aliasing:

            col("a").over(col("c"))
            col("a").over(col("d"))
            col("a").over(col("e"))

        This would also cause an error if we needed to expand both sides:

            col("a", "b").over(col("c", "d", "e"))

        Since that would become:

            col("a").over(col("c"))
            col("b").over(col("d"))
            col(<MISSING>).over(col("e"))  # InvalidOperationError: cannot combine selectors that produce a different number of columns (3 != 2)
        """
        iterable = (child.iter_expand(self) for child in children)
        return tuple(e for sub in iterable for e in sub)

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
