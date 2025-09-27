"""Sugar for working with [Acero].

[`pyarrow.acero`] has some building blocks for constructing queries, but is
quite verbose when used directly.

This module aligns some apis to look more like `polars`.

[Acero]: https://arrow.apache.org/docs/cpp/acero/overview.html
[`pyarrow.acero`]: https://arrow.apache.org/docs/python/api/acero.html
"""

from __future__ import annotations

import operator
from functools import reduce
from itertools import chain
from typing import TYPE_CHECKING, Any, Literal, TypeVar, Union

import pyarrow as pa  # ignore-banned-import
import pyarrow.acero as pac
import pyarrow.compute as pc  # ignore-banned-import
from pyarrow.acero import Declaration as Decl

from narwhals.typing import SingleColSelector

if TYPE_CHECKING:
    from collections.abc import Iterable

    from typing_extensions import TypeAlias

    from narwhals._arrow.typing import (  # type: ignore[attr-defined]
        AggregateOptions as _AggregateOptions,
        Aggregation as _Aggregation,
    )
    from narwhals._plan.typing import Seq
    from narwhals.typing import NonNestedLiteral

T = TypeVar("T")
OneOrListOrTuple: TypeAlias = Union[T, list[T], tuple[T, ...]]
"""WARNING: Don't use this unless there is a runtime check for exactly `list | tuple`."""


Incomplete: TypeAlias = Any
Expr: TypeAlias = pc.Expression
IntoExpr: TypeAlias = "Expr | NonNestedLiteral"
Field: TypeAlias = Union[Expr, SingleColSelector]
"""Anything that passes as a single item in [`_compute._ensure_field_ref`].

[`_compute._ensure_field_ref`]: https://github.com/apache/arrow/blob/9b96bdbc733d62f0375a2b1b9806132abc19cd3f/python/pyarrow/_compute.pyx#L1507-L1531
"""

AggKeys: TypeAlias = "Iterable[Field] | None"

Target: TypeAlias = OneOrListOrTuple[Field]
Aggregation: TypeAlias = "_Aggregation"
AggregateOptions: TypeAlias = "_AggregateOptions"
Opts: TypeAlias = "AggregateOptions | None"
OutputName: TypeAlias = str
AggSpec: TypeAlias = tuple[Target, Aggregation, Opts, OutputName]


# TODO @dangotbanned: Rename
def pc_expr(into: IntoExpr, /, *, str_as_lit: bool = False) -> Expr:
    if isinstance(into, pc.Expression):
        return into
    if isinstance(into, str) and not str_as_lit:
        return pc.field(into)
    arg: Incomplete = into
    return pc.scalar(arg)


def _parse_all_horizontal(predicates: Seq[Expr], constraints: dict[str, Any], /) -> Expr:
    if not constraints and len(predicates) == 1:
        return predicates[0]
    it = (
        pc.field(name) == pc_expr(v, str_as_lit=True) for name, v in constraints.items()
    )
    return reduce(operator.and_, chain(predicates, it))


# TODO @dangotbanned: Docs (currently copy/paste from `pyarrow`)
def table_source(native: pa.Table, /) -> Decl:
    """A Source node which accepts a table."""
    return Decl("table_source", options=pac.TableSourceNodeOptions(native))


def _aggregate(agg_specs: Iterable[AggSpec], /, keys: AggKeys = None) -> Decl:
    # NOTE: See https://github.com/apache/arrow/blob/9b96bdbc733d62f0375a2b1b9806132abc19cd3f/python/pyarrow/_acero.pyx#L167-L192
    aggs: Incomplete = agg_specs
    keys_: Incomplete = keys
    return Decl("aggregate", pac.AggregateNodeOptions(aggs, keys=keys_))


# TODO @dangotbanned: Plan
# TODO @dangotbanned: Docs (currently copy/paste from `pyarrow`)
def aggregate(aggs: Iterable[AggSpec], /) -> Decl:
    """Scalar aggregate.

    Reduce an array or scalar input to a single scalar output (e.g. computing the mean of a column)
    """
    return _aggregate(aggs)


# TODO @dangotbanned: Docs (currently copy/paste from `pyarrow`)
def group_by(keys: AggKeys, aggs: Iterable[AggSpec], /) -> Decl:
    """Hash aggregate.

    Like GROUP BY in SQL and first partition data based on one or more key columns,
    then reduce the data in each partition.
    """
    return _aggregate(aggs, keys=keys)


def filter(*predicates: Expr, **constraints: IntoExpr) -> Decl:
    """Selects rows where all expressions evaluate to True.

    Arguments:
        predicates: [`Expression`](s) which must all have a return type of boolean.
        constraints: Column filters; use `name = value` to filter columns by the supplied value.

    Notes:
        - Uses logic similar to [`polars`] for an AND-reduction
        - Elements where the filter does not evaluate to True are discarded, **including nulls**

    [`Expression`]: https://arrow.apache.org/docs/python/generated/pyarrow.dataset.Expression.html
    [`polars`]: https://github.com/pola-rs/polars/blob/d0914d416ce4e1dfcb5f946875ffd1181e31c493/py-polars/polars/_utils/parse/expr.py#L199-L242
    """
    expr = _parse_all_horizontal(predicates, constraints)
    return Decl("filter", options=pac.FilterNodeOptions(expr))


# TODO @dangotbanned: Plan
def select(*exprs: IntoExpr, **named_exprs: IntoExpr) -> Decl:
    raise NotImplementedError


# TODO @dangotbanned: Docs (currently copy/paste from `pyarrow`)
def project(**named_exprs: Expr) -> Decl:
    """Make a node which executes expressions on input batches, producing batches of the same length with new columns.

    This is the option class for the "project" node factory.

    The "project" operation rearranges, deletes, transforms, and
    creates columns. Each output column is computed by evaluating
    an expression against the source record batch. These must be
    scalar expressions (expressions consisting of scalar literals,
    field references and scalar functions, i.e. elementwise functions
    that return one value for each input row independent of the value
    of all other rows).
    """
    # NOTE: Both just need to be sized and iterable
    names: Incomplete = named_exprs.keys()
    exprs: Incomplete = named_exprs.values()
    return Decl("project", options=pac.ProjectNodeOptions(exprs, names))


# TODO @dangotbanned: Find which option class this uses
def order_by(
    sort_keys: tuple[tuple[str, Literal["ascending", "descending"]], ...] = (),
    *,
    null_placement: Literal["at_start", "at_end"] = "at_end",
) -> Decl:
    return Decl(
        "order_by", pac.OrderByNodeOptions(sort_keys, null_placement=null_placement)
    )


# TODO @dangotbanned: Docs
def collect(*declarations: Decl, use_threads: bool = True) -> pa.Table:
    # NOTE: stubs + docs say `list`, but impl allows any iterable
    decls: Incomplete = declarations
    return Decl.from_sequence(decls).to_table(use_threads=use_threads)


# NOTE: Composite functions are suffixed with `_table`
def group_by_table(
    native: pa.Table, keys: AggKeys, aggs: Iterable[AggSpec], *, use_threads: bool
) -> pa.Table:
    """Adapted from [`pa.TableGroupBy.aggregate`] and [`pa.acero._group_by`].

    - Backport of [apache/arrow#36768].
      - `first` and `last` were [broken in `pyarrow==13`].
    - Also allows us to specify our own aliases for aggregate output columns.
      - Fixes [narwhals-dev/narwhals#1612]

    [`pa.TableGroupBy.aggregate`]: https://github.com/apache/arrow/blob/0e7e70cfdef4efa287495272649c071a700c34fa/python/pyarrow/table.pxi#L6600-L6626
    [`pa.acero._group_by`]: https://github.com/apache/arrow/blob/0e7e70cfdef4efa287495272649c071a700c34fa/python/pyarrow/acero.py#L412-L418
    [apache/arrow#36768]: https://github.com/apache/arrow/pull/36768
    [broken in `pyarrow==13`]: https://github.com/apache/arrow/issues/36709
    [narwhals-dev/narwhals#1612]: https://github.com/narwhals-dev/narwhals/issues/1612
    """
    return collect(table_source(native), group_by(keys, aggs), use_threads=use_threads)


# TODO @dangotbanned: Docs?
def filter_table(native: pa.Table, *predicates: Expr, **constraints: Any) -> pa.Table:
    return collect(table_source(native), filter(*predicates, **constraints))
