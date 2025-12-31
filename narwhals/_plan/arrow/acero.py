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
from typing import TYPE_CHECKING, Any, Final, Literal, Union, cast

import pyarrow as pa  # ignore-banned-import
import pyarrow.acero as pac
import pyarrow.compute as pc  # ignore-banned-import
from pyarrow.acero import Declaration as Decl

from narwhals._plan.common import ensure_list_str, temp
from narwhals._plan.typing import NonCrossJoinStrategy, OneOrSeq
from narwhals._utils import check_column_names_are_unique
from narwhals.typing import AsofJoinStrategy, JoinStrategy, SingleColSelector

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
        ChunkedArrayAny,
        JoinTypeSubset,
        ScalarAny,
    )
    from narwhals._plan.typing import OneOrIterable, Seq
    from narwhals.typing import NonNestedLiteral

Incomplete: TypeAlias = Any
Expr: TypeAlias = pc.Expression
IntoExpr: TypeAlias = "Expr | NonNestedLiteral | ScalarAny"
Field: TypeAlias = Union[Expr, SingleColSelector]
"""Anything that passes as a single item in [`_compute._ensure_field_ref`].

[`_compute._ensure_field_ref`]: https://github.com/apache/arrow/blob/9b96bdbc733d62f0375a2b1b9806132abc19cd3f/python/pyarrow/_compute.pyx#L1507-L1531
"""

Target: TypeAlias = OneOrSeq[Field]
Aggregation: TypeAlias = Union[
    "_Aggregation",
    Literal[
        "hash_kurtosis",
        "hash_skew",
        "hash_pivot_wider",
        "kurtosis",
        "skew",
        "pivot_wider",
    ],
]
AggregateOptions: TypeAlias = "_AggregateOptions"
Opts: TypeAlias = "AggregateOptions | None"
OutputName: TypeAlias = str

IntoDecl: TypeAlias = Union[pa.Table, Decl]
"""An in-memory table, or a plan that began with one."""

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


def _into_decl(source: IntoDecl, /) -> Decl:
    return source if not isinstance(source, pa.Table) else table_source(source)


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


def _add_column(native: pa.Table, index: int, name: str, values: IntoExpr) -> Decl:
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
    return declare(table_source(native), _project(exprs, names))


def append_column(native: pa.Table, name: str, values: IntoExpr) -> Decl:
    return _add_column(native, native.num_columns, name, values)


def prepend_column(native: pa.Table, name: str, values: IntoExpr) -> Decl:
    return _add_column(native, 0, name, values)


def _join_options(
    how: NonCrossJoinStrategy,
    left_on: OneOrIterable[str],
    right_on: OneOrIterable[str],
    suffix: str = "_right",
    left_names: Iterable[str] | None = None,
    right_names: Iterable[str] = (),
    *,
    coalesce_keys: bool = True,
) -> pac.HashJoinNodeOptions:
    right_on = ensure_list_str(right_on)
    rhs_names: Iterable[str] | None = None
    # polars full join does not coalesce keys
    if not (coalesce_keys and (how != "full")):
        lhs_names = None
    else:
        lhs_names = left_names
        if how in {"inner", "left"}:
            rhs_names = (name for name in right_names if name not in right_on)
    tp: Incomplete = pac.HashJoinNodeOptions
    return tp(  # type: ignore[no-any-return]
        _HOW_JOIN[how],
        left_keys=ensure_list_str(left_on),
        right_keys=right_on,
        left_output=lhs_names,
        right_output=rhs_names,
        output_suffix_for_right=suffix,
    )


def _hashjoin(
    left: IntoDecl, right: IntoDecl, /, options: pac.HashJoinNodeOptions
) -> Decl:
    return Decl("hashjoin", options, [_into_decl(left), _into_decl(right)])


def _asofjoin(
    left: pa.Table, right: pa.Table, /, options: pac.AsofJoinNodeOptions
) -> Decl:
    return Decl("asofjoin", options, [table_source(left), table_source(right)])


def _join_asof_ensure_no_collisions(
    left: pa.Table,
    right: pa.Table,
    right_on: str,
    right_by: Sequence[str] = (),
    suffix: str = "_right",
) -> pa.Table:
    """Adapted from [upstream] to avoid raising early.

    [upstream]: https://github.com/apache/arrow/blob/9b03118e834dfdaa0cf9e03595477b499252a9cb/python/pyarrow/acero.py#L306-L316
    """
    right_names = right.schema.names
    allowed = {right_on, *right_by}
    if collisions := set(right_names).difference(allowed).intersection(left.schema.names):
        renamed = [f"{nm}{suffix}" if nm in collisions else nm for nm in right_names]
        return right.rename_columns(renamed)
    return right


def _join_asof_strategy_to_tolerance(
    left_on: ChunkedArrayAny, right_on: ChunkedArrayAny, /, strategy: AsofJoinStrategy
) -> int:
    """Calculate the **required** `tolerance` argument, from `*on` values and strategy.

    For both `polars` and `pandas` this is optional and in `narwhals` it isn't supported (yet).

    So we need to get the lowest/highest value for a match and use that for a similar default.

    `"backward"`:

        tolerance <= right_on - left_on <= 0

    `"forward"`:

        0 <= right_on - left_on <= tolerance

    Note:
        `tolerance` is interpreted in the same units as the `on` keys.
    """
    import narwhals._plan.arrow.functions as fn

    if strategy == "nearest":
        msg = "Only 'backward' and 'forward' strategies are currently supported for `pyarrow`"
        raise NotImplementedError(msg)
    lower = fn.min_horizontal(fn.min_(left_on), fn.min_(right_on))
    upper = fn.max_horizontal(fn.max_(left_on), fn.max_(right_on))
    scalar = fn.sub(lower, upper) if strategy == "backward" else fn.sub(upper, lower)
    tolerance: int = fn.cast(scalar, fn.I64).as_py()
    return tolerance


def join_asof_tables(
    left: pa.Table,
    right: pa.Table,
    left_on: str,
    right_on: str,
    *,
    left_by: Sequence[str] = (),
    right_by: Sequence[str] = (),
    strategy: AsofJoinStrategy = "backward",
    suffix: str = "_right",
) -> pa.Table:
    right = _join_asof_ensure_no_collisions(
        left, right, right_on, right_by, suffix=suffix
    )
    tolerance = _join_asof_strategy_to_tolerance(
        left.column(left_on), right.column(right_on), strategy
    )
    lb: list[Any] = [] if not left_by else list(left_by)
    rb: list[Any] = [] if not right_by else list(right_by)
    join_opts = pac.AsofJoinNodeOptions(
        left_on=left_on, right_on=right_on, left_by=lb, right_by=rb, tolerance=tolerance
    )
    return _asofjoin(left, right, join_opts).to_table()


def declare(*declarations: Decl) -> Decl:
    """Compose one or more `Declaration` nodes for execution as a pipeline."""
    if len(declarations) == 1:
        return declarations[0]
    # NOTE: stubs + docs say `list`, but impl allows any iterable
    decls: Incomplete = declarations
    return Decl.from_sequence(decls)


def collect(
    *declarations: Decl,
    use_threads: bool = True,
    ensure_unique_column_names: bool = False,
) -> pa.Table:
    """Compose and evaluate a logical plan.

    Arguments:
        *declarations: One or more `Declaration` nodes to execute as a pipeline.
            **The first node must be a `table_source`**.
        use_threads: Pass `False` if `declarations` contains any order-dependent aggregation(s).
        ensure_unique_column_names: Pass `True` if `declarations` adds generated column names that were
            not explicitly defined on the `narwhals`-side. E.g. `join(suffix=...)`.
    """
    result = declare(*declarations).to_table(use_threads=use_threads)
    if ensure_unique_column_names:
        check_column_names_are_unique(result.column_names)
    return result


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
    how: NonCrossJoinStrategy,
    left_on: OneOrIterable[str],
    right_on: OneOrIterable[str],
    suffix: str = "_right",
    *,
    coalesce_keys: bool = True,
) -> pa.Table:
    """Join two tables.

    Based on:
    - [`pyarrow.Table.join`]
    - [`pyarrow.acero._perform_join`]
    - [`narwhals._arrow.dataframe.DataFrame.join`]

    [`pyarrow.Table.join`]: https://github.com/apache/arrow/blob/f7320c9a40082639f9e0cf8b3075286e3fc6c0b9/python/pyarrow/table.pxi#L5764-L5772
    [`pyarrow.acero._perform_join`]: https://github.com/apache/arrow/blob/f7320c9a40082639f9e0cf8b3075286e3fc6c0b9/python/pyarrow/acero.py#L82-L260
    [`narwhals._arrow.dataframe.DataFrame.join`]: https://github.com/narwhals-dev/narwhals/blob/f4787d3f9e027306cb1786db7b471f63b393b8d1/narwhals/_arrow/dataframe.py#L393-L433
    """
    left_on = left_on or ()
    right_on = right_on or left_on
    opts = _join_options(
        how,
        left_on,
        right_on,
        suffix,
        left.schema.names,
        right.schema.names,
        coalesce_keys=coalesce_keys,
    )
    return collect(_hashjoin(left, right, opts), ensure_unique_column_names=True)


# TODO @dangotbanned: Adapt this into a `Declaration` that handles more of `ArrowGroupBy.agg_over`
def join_inner_tables(left: pa.Table, right: pa.Table, on: list[str]) -> pa.Table:
    """Fast path for use with `over`.

    Has almost zero branching and the bodys of helper functions are inlined.

    Eventually want to adapt this into:

        goal = declare(
            join_inner(
                declare(table_source(compliant.native), select_names(key_names)),
                declare(table_source(ordered), group_by(key_names, specs)),
            ),
            select_names(agg_names),
        )
    """
    tp: Incomplete = pac.HashJoinNodeOptions
    opts = tp(
        "inner",
        left_keys=on,
        right_keys=on,
        left_output=left.schema.names,
        right_output=(name for name in right.schema.names if name not in on),
        output_suffix_for_right="_right",
    )
    lhs, rhs = pac.TableSourceNodeOptions(left), pac.TableSourceNodeOptions(right)
    decl = Decl("hashjoin", opts, [Decl("table_source", lhs), Decl("table_source", rhs)])
    result = decl.to_table()
    check_column_names_are_unique(result.column_names)
    return result


def join_cross_tables(
    left: pa.Table, right: pa.Table, suffix: str = "_right", *, coalesce_keys: bool = True
) -> pa.Table:
    """Perform a cross join between tables."""
    left_names, right_names = left.column_names, right.column_names
    on = temp.column_name(set().union(left_names, right_names))
    opts = _join_options(
        how="inner",
        left_on=on,
        right_on=on,
        suffix=suffix,
        left_names=[on, *left_names],
        right_names=right_names,
        coalesce_keys=coalesce_keys,
    )
    left_, right_ = prepend_column(left, on, 0), prepend_column(right, on, 0)
    decl = _hashjoin(left_, right_, opts)
    return collect(decl, ensure_unique_column_names=True).remove_column(0)


def _add_column_table(
    native: pa.Table, index: int, name: str, values: IntoExpr | ArrowAny
) -> pa.Table:
    if isinstance(values, (pa.ChunkedArray, pa.Array)):
        return native.add_column(index, name, values)
    return _add_column(native, index, name, values).to_table()


def append_column_table(
    native: pa.Table, name: str, values: IntoExpr | ArrowAny
) -> pa.Table:
    return _add_column_table(native, native.num_columns, name, values)


def prepend_column_table(
    native: pa.Table, name: str, values: IntoExpr | ArrowAny
) -> pa.Table:
    return _add_column_table(native, 0, name, values)
