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

from narwhals._plan.common import ensure_list_str, flatten_hash_safe, temp
from narwhals._plan.options import SortMultipleOptions
from narwhals._plan.typing import OneOrSeq
from narwhals.typing import JoinStrategy, SingleColSelector

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Collection,
        Iterable,
        Iterator,
        Mapping,
        Sequence,
    )

    from typing_extensions import TypeAlias, TypeIs

    from narwhals._arrow.typing import (  # type: ignore[attr-defined]
        AggregateOptions as _AggregateOptions,
        Aggregation as _Aggregation,
    )
    from narwhals._plan.arrow.group_by import AggSpec
    from narwhals._plan.arrow.typing import (
        ArrowAny,
        JoinTypeSubset,
        NullPlacement,
        ScalarAny,
    )
    from narwhals._plan.typing import OneOrIterable, Order, Seq
    from narwhals.typing import NonNestedLiteral

Incomplete: TypeAlias = Any
Expr: TypeAlias = pc.Expression
IntoExpr: TypeAlias = "Expr | NonNestedLiteral | ScalarAny"
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
lit = cast("Callable[[NonNestedLiteral | ScalarAny], Expr]", pc.scalar)
"""Alias for `pyarrow.compute.scalar`.

Extends the signature from `bool | float | str`.

See https://github.com/apache/arrow/pull/47609#discussion_r2392499842
"""

_HOW_JOIN: Mapping[JoinStrategy, JoinTypeSubset] = {
    "inner": "inner",
    "left": "left outer",
    "full": "full outer",
    "anti": "left anti",
    "semi": "left semi",
}


# NOTE: ATOW there are 304 valid function names, 46 can be used for some kind of agg
# Due to expr expansion, it is very likely that we have repeat runs
@functools.lru_cache(maxsize=128)
def can_thread(function_name: str, /) -> bool:
    return function_name not in _THREAD_UNSAFE


def cols_iter(names: Iterable[str], /) -> Iterator[Expr]:
    for name in names:
        yield col(name)


def _is_expr(obj: Any) -> TypeIs[pc.Expression]:
    return isinstance(obj, pc.Expression)


def _parse_into_expr(into: IntoExpr, /, *, str_as_lit: bool = False) -> Expr:
    if _is_expr(into):
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


def aggregate(aggs: Iterable[AggSpec], /) -> Decl:
    """May only use [Scalar aggregate] functions.

    [Scalar aggregate]: https://arrow.apache.org/docs/cpp/compute.html#aggregations
    """
    return _aggregate(aggs)


def group_by(keys: Iterable[Field], aggs: Iterable[AggSpec], /) -> Decl:
    """May only use [Hash aggregate] functions, requires grouping.

    [Hash aggregate]: https://arrow.apache.org/docs/cpp/compute.html#grouped-aggregations-group-by
    """
    return _aggregate(aggs, keys=keys)


def filter(*predicates: Expr, **constraints: IntoExpr) -> Decl:
    expr = _parse_all_horizontal(predicates, constraints)
    return Decl("filter", options=pac.FilterNodeOptions(expr))


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


def _add_column(
    native: pa.Table, index: int, name: str, values: IntoExpr | ArrowAny
) -> pa.Table:
    if isinstance(values, (pa.ChunkedArray, pa.Array)):
        return native.add_column(index, name, values)
    column = values if _is_expr(values) else lit(values)
    schema = native.schema
    schema_names = schema.names
    if index == 0:
        names: Sequence[str] = (name, *schema_names)
        exprs = (column, *cols_iter(schema_names))
    elif index == native.num_columns:
        names = (*schema_names, name)
        exprs = (*cols_iter(schema_names), column)
    else:
        schema_names.insert(index, name)
        names = schema_names
        exprs = tuple(_parse_into_iter_expr(nm if nm != name else column for nm in names))
    return collect(table_source(native), _project(exprs, names))


def append_column(native: pa.Table, name: str, values: IntoExpr | ArrowAny) -> pa.Table:
    return _add_column(native, native.num_columns, name, values)


def prepend_column(native: pa.Table, name: str, values: IntoExpr | ArrowAny) -> pa.Table:
    return _add_column(native, 0, name, values)


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


def sort_by(
    by: OneOrIterable[str],
    *more_by: str,
    descending: OneOrIterable[bool] = False,
    nulls_last: bool = False,
) -> Decl:
    return SortMultipleOptions.parse(
        descending=descending, nulls_last=nulls_last
    ).to_arrow_acero(tuple(flatten_hash_safe((by, more_by))))


def join(
    left: pa.Table,
    right: pa.Table,
    how: JoinTypeSubset,
    left_on: OneOrIterable[str],
    right_on: OneOrIterable[str],
    suffix: str = "_right",
    *,
    coalesce_keys: bool = True,
) -> Decl:
    """Heavily based on [`pyarrow.acero._perform_join`].

    [`pyarrow.acero._perform_join`]: https://github.com/apache/arrow/blob/f7320c9a40082639f9e0cf8b3075286e3fc6c0b9/python/pyarrow/acero.py#L82-L260
    """
    left_on = ensure_list_str(left_on)
    right_on = ensure_list_str(right_on)

    # polars full join does not coalesce keys,
    coalesce_keys = coalesce_keys and (how != "full outer")
    if not coalesce_keys:
        opts = _join_options(how, left_on, right_on, suffix=suffix)
        return _hashjoin(left, right, opts)

    # By default expose all columns on both left and right table
    left_names = left.schema.names
    right_names = right.schema.names

    if how in {"left semi", "left anti"}:
        right_names = []
    elif how in {"inner", "left outer"}:
        right_names = [name for name in right_names if name not in right_on]
    opts = _join_options(
        how,
        left_on,
        right_on,
        suffix=suffix,
        left_output=left_names,
        right_output=right_names,
    )
    return _hashjoin(left, right, opts)


def _join_options(
    how: JoinTypeSubset,
    left_on: str | list[str],
    right_on: str | list[str],
    *,
    suffix: str = "_right",
    left_output: Iterable[str] | None = None,
    right_output: Iterable[str] | None = None,
) -> pac.HashJoinNodeOptions:
    tp: Incomplete = pac.HashJoinNodeOptions
    kwds = {
        "left_output": left_output,
        "right_output": right_output,
        "output_suffix_for_right": suffix,
    }
    return tp(how, left_on, right_on, **kwds)  # type: ignore[no-any-return]


def _hashjoin(
    left: pa.Table, right: pa.Table, /, options: pac.HashJoinNodeOptions
) -> Decl:
    return Decl("hashjoin", options, [table_source(left), table_source(right)])


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


def join_tables(
    left: pa.Table,
    right: pa.Table,
    how: JoinStrategy,
    left_on: OneOrIterable[str] | None = (),
    right_on: OneOrIterable[str] | None = (),
    suffix: str = "_right",
    *,
    coalesce_keys: bool = True,
) -> pa.Table:
    if how == "cross":
        return _join_cross_tables(left, right, suffix)
    join_type = _HOW_JOIN[how]
    left_on = left_on or ()
    right_on = right_on or left_on
    decl = join(
        left, right, join_type, left_on, right_on, suffix, coalesce_keys=coalesce_keys
    )
    return collect(decl)


# TODO @dangotbanned: Very rough start to get tests passing
# - Decouple from `pa.Table` & collecting 3 times
# - Reuse the plan from `_add_column`
# - Write some more specialized parsers for
#   [x] column names
#   [ ] indices?
def _join_cross_tables(
    left: pa.Table, right: pa.Table, suffix: str = "_right"
) -> pa.Table:
    key_token = temp.column_name(chain(left.column_names, right.column_names))
    result = join_tables(
        prepend_column(left, key_token, 0),
        prepend_column(right, key_token, 0),
        how="inner",
        left_on=key_token,
        suffix=suffix,
    )
    return result.remove_column(0)
