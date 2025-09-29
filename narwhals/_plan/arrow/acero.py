"""Sugar for working with [Acero].

[`pyarrow.acero`] has some building blocks for constructing queries, but is
quite verbose when used directly.

This module aligns some apis to look more like `polars`.

Notes:
    - Functions suffixed with `_table` all handle composition and collection internally

[Acero]: https://arrow.apache.org/docs/cpp/acero/overview.html
[`pyarrow.acero`]: https://arrow.apache.org/docs/python/api/acero.html
"""

from __future__ import annotations

import functools
import operator
from functools import reduce
from itertools import chain
from typing import TYPE_CHECKING, Any, Final, Union, cast

import pyarrow as pa  # ignore-banned-import
import pyarrow.acero as pac
import pyarrow.compute as pc  # ignore-banned-import
from pyarrow.acero import Declaration as Decl

from narwhals._plan.typing import OneOrSeq
from narwhals.typing import SingleColSelector

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterable, Iterator

    from typing_extensions import TypeAlias

    from narwhals._arrow.typing import (  # type: ignore[attr-defined]
        AggregateOptions as _AggregateOptions,
        Aggregation as _Aggregation,
    )
    from narwhals._plan.arrow.group_by import AggSpec
    from narwhals._plan.arrow.typing import NullPlacement
    from narwhals._plan.typing import OneOrIterable, Order, Seq
    from narwhals.typing import NonNestedLiteral

Incomplete: TypeAlias = Any
Expr: TypeAlias = pc.Expression
IntoExpr: TypeAlias = "Expr | NonNestedLiteral"
Field: TypeAlias = Union[Expr, SingleColSelector]
"""Anything that passes as a single item in [`_compute._ensure_field_ref`].

[`_compute._ensure_field_ref`]: https://github.com/apache/arrow/blob/9b96bdbc733d62f0375a2b1b9806132abc19cd3f/python/pyarrow/_compute.pyx#L1507-L1531
"""

Target: TypeAlias = OneOrSeq[Field]
Aggregation: TypeAlias = "_Aggregation"
AggregateOptions: TypeAlias = "_AggregateOptions"
Opts: TypeAlias = "AggregateOptions | None"
OutputName: TypeAlias = str

_THREAD_UNSAFE: Final = frozenset[Aggregation](
    ("hash_first", "hash_last", "first", "last")
)
col = pc.field
lit = cast("Callable[[NonNestedLiteral], Expr]", pc.scalar)
"""Alias for `pyarrow.compute.scalar`."""


# NOTE: ATOW there are 304 valid function names, 46 can be used for some kind of agg
# Due to expr expansion, it is very likely that we have repeat runs
@functools.lru_cache(maxsize=128)
def can_thread(function_name: str, /) -> bool:
    return function_name not in _THREAD_UNSAFE


def _parse_into_expr(into: IntoExpr, /, *, str_as_lit: bool = False) -> Expr:
    if isinstance(into, pc.Expression):
        return into
    if isinstance(into, str) and not str_as_lit:
        return col(into)
    return lit(into)


def _parse_into_iter_expr(inputs: Iterable[IntoExpr], /) -> Iterator[Expr]:
    for into_expr in inputs:
        yield _parse_into_expr(into_expr)


def _parse_into_seq_of_expr(inputs: Iterable[IntoExpr], /) -> Seq[Expr]:
    return tuple(_parse_into_iter_expr(inputs))


def _parse_all_horizontal(predicates: Seq[Expr], constraints: dict[str, Any], /) -> Expr:
    if not constraints and len(predicates) == 1:
        return predicates[0]
    it = (
        col(name) == _parse_into_expr(v, str_as_lit=True)
        for name, v in constraints.items()
    )
    return reduce(operator.and_, chain(predicates, it))


def table_source(native: pa.Table, /) -> Decl:
    """Start building a logical plan, using `native` as the source table.

    All calls to `collect` must use this as the first `Declaration`.
    """
    return Decl("table_source", options=pac.TableSourceNodeOptions(native))


def _aggregate(aggs: Iterable[AggSpec], /, keys: Iterable[Field] | None = None) -> Decl:
    # NOTE: See https://github.com/apache/arrow/blob/9b96bdbc733d62f0375a2b1b9806132abc19cd3f/python/pyarrow/_acero.pyx#L167-L192
    aggs_: Incomplete = aggs
    keys_: Incomplete = keys
    return Decl("aggregate", pac.AggregateNodeOptions(aggs_, keys=keys_))


# TODO @dangotbanned: Plan
# TODO @dangotbanned: Docs (currently copy/paste from `pyarrow`)
def aggregate(aggs: Iterable[AggSpec], /) -> Decl:
    """Scalar aggregate.

    Reduce an array or scalar input to a single scalar output (e.g. computing the mean of a column)
    """
    return _aggregate(aggs)


# TODO @dangotbanned: Docs (currently copy/paste from `pyarrow`)
def group_by(keys: Iterable[Field], aggs: Iterable[AggSpec], /) -> Decl:
    """Hash aggregate.

    Like GROUP BY in SQL and first partition data based on one or more key columns,
    then reduce the data in each partition.
    """
    return _aggregate(aggs, keys=keys)


def filter(*predicates: Expr, **constraints: IntoExpr) -> Decl:
    expr = _parse_all_horizontal(predicates, constraints)
    return Decl("filter", options=pac.FilterNodeOptions(expr))


# TODO @dangotbanned: Plan
def select(*exprs: IntoExpr, **named_exprs: IntoExpr) -> Decl:
    raise NotImplementedError


def select_names(column_names: OneOrIterable[str], *more_names: str) -> Decl:
    """`select` where all args are column names."""
    if not more_names:
        if isinstance(column_names, str):
            return _project((col(column_names),), (column_names,))
        more_names = tuple(column_names)
    elif isinstance(column_names, str):
        more_names = column_names, *more_names
    else:
        msg = f"Passing both iterable and positional inputs is not supported.\n{column_names=}\n{more_names=}"
        raise NotImplementedError(msg)
    return _project([col(name) for name in more_names], more_names)


def _project(exprs: Collection[Expr], names: Collection[str]) -> Decl:
    # NOTE: Both just need to be `Sized` and `Iterable`
    exprs_: Incomplete = exprs
    names_: Incomplete = names
    return Decl("project", options=pac.ProjectNodeOptions(exprs_, names_))


def project(**named_exprs: IntoExpr) -> Decl:
    """Similar to `select`, but more rigid.

    Arguments:
        **named_exprs: Inputs composed of any combination of

            - Column names or `pc.field`
            - Python literals or `pc.scalar` (for `str` literals)
            - [Scalar functions] applied to the above

    Notes:
        - [`Expression`]s have no concept of aliasing, therefore, all inputs must be `**named_exprs`.
        - Always returns a table with the same length, scalar literals are broadcast unconditionally.

    [`Expression`]: https://arrow.apache.org/docs/python/generated/pyarrow.dataset.Expression.html
    [Scalar functions]: https://arrow.apache.org/docs/cpp/compute.html#element-wise-scalar-functions
    """
    exprs = _parse_into_seq_of_expr(named_exprs.values())
    return _project(names=named_exprs.keys(), exprs=exprs)


def _order_by(
    sort_keys: Iterable[tuple[str, Order]] = (),
    *,
    null_placement: NullPlacement = "at_end",
) -> Decl:
    # NOTE: There's no runtime type checking of `sort_keys` wrt shape
    # Just need to be `Iterable`and unpack like a 2-tuple
    # https://github.com/apache/arrow/blob/9b96bdbc733d62f0375a2b1b9806132abc19cd3f/python/pyarrow/_compute.pyx#L77-L88
    keys: Incomplete = sort_keys
    return Decl("order_by", pac.OrderByNodeOptions(keys, null_placement=null_placement))


# TODO @dangotbanned: Utilize `SortMultipleOptions.to_arrow_acero`
def sort_by(*args: Any, **kwds: Any) -> Decl:
    msg = "Should convert from polars args -> use `_order_by"
    raise NotImplementedError(msg)


def collect(*declarations: Decl, use_threads: bool = True) -> pa.Table:
    """Compose and evaluate a logical plan.

    Arguments:
        *declarations: One or more `Declaration` nodes to execute as a pipeline.
            **The first node must be a `table_source`**.
        use_threads: Pass `False` if `declarations` contains any order-dependent aggregation(s).
    """
    # NOTE: stubs + docs say `list`, but impl allows any iterable
    decls: Incomplete = declarations
    return Decl.from_sequence(decls).to_table(use_threads=use_threads)


def group_by_table(
    native: pa.Table, keys: Iterable[Field], aggs: Iterable[AggSpec]
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
    aggs = tuple(aggs)
    use_threads = all(spec.use_threads for spec in aggs)
    return collect(table_source(native), group_by(keys, aggs), use_threads=use_threads)


def filter_table(native: pa.Table, *predicates: Expr, **constraints: Any) -> pa.Table:
    """Selects rows where all expressions evaluate to True.

    Arguments:
        native: source table
        predicates: [`Expression`]s which must all have a return type of boolean.
        constraints: Column filters; use `name = value` to filter columns by the supplied value.

    Notes:
        - Uses logic similar to [`polars`] for an AND-reduction
        - Elements where the filter does not evaluate to True are discarded, **including nulls**

    [`Expression`]: https://arrow.apache.org/docs/python/generated/pyarrow.dataset.Expression.html
    [`polars`]: https://github.com/pola-rs/polars/blob/d0914d416ce4e1dfcb5f946875ffd1181e31c493/py-polars/polars/_utils/parse/expr.py#L199-L242
    """
    return collect(table_source(native), filter(*predicates, **constraints))


def select_names_table(
    native: pa.Table, column_names: OneOrIterable[str], *more_names: str
) -> pa.Table:
    return collect(table_source(native), select_names(column_names, *more_names))
