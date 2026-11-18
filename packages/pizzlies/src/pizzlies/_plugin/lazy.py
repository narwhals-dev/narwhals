"""Lazy frames for `pizzlies`, with projection pushdown.

A `DictLazyFrame` doesn't compute anything: each method records an operation in a
`LazyPlan`, whose leaves are in-memory frames or Parquet scans. `collect` first prunes
the columns which nothing downstream reads, so that scans only decode the columns a
query uses, and then replays the plan with `DictDataFrame`.

Column usage is read off Narwhals' expression metadata. Whenever it's unclear which
columns an operation reads (e.g. `all()`, selectors, `unpivot`), it keeps them all.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from narwhals.compliant import CompliantGroupBy, CompliantLazyFrame
from narwhals.utils import Implementation
from pizzlies._plugin._vendored import not_implemented
from pizzlies._plugin.dataframe import DictDataFrame
from pizzlies._plugin.parquet import read_parquet
from pizzlies._plugin.repr import fit_lines
from pizzlies._plugin.series import DictSeries

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
    from types import ModuleType

    from typing_extensions import Self, TypeIs

    from narwhals import LazyFrame
    from narwhals.dtypes import DType
    from narwhals.typing import AsofJoinStrategy, ConcatMethod, IntoDType, JoinStrategy
    from narwhals.utils import Version
    from pizzlies._plugin.expr import DictExpr
    from pizzlies._plugin.namespace import DictNamespace
    from pizzlies._plugin.typing import Context, DictFrame

    Columns = frozenset[str] | None
    """The columns a plan must provide, or `None` for all of them."""


# --- Plans ---------------------------------------------------------------------------


class LazyPlan:
    """A query plan: the native object of `pizzlies`' lazy frames. Immutable."""

    __slots__ = ()
    inputs: tuple[LazyPlan, ...] = ()

    def explain(self) -> str:
        """The plan as an indented tree, operations above their inputs."""
        lines: list[str] = []

        def visit(plan: LazyPlan, depth: int) -> None:
            lines.append(f"{'  ' * depth}{plan._describe()}")
            for child in plan.inputs:
                visit(child, depth + 1)

        visit(self, 0)
        return "\n".join(lines)

    def _describe(self) -> str:  # pragma: no cover
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"LazyPlan:\n{self.explain()}"


class FramePlan(LazyPlan):
    """An in-memory dict of lists."""

    __slots__ = ("native",)

    def __init__(self, native: DictFrame) -> None:
        self.native = native

    def _describe(self) -> str:
        return f"FRAME {list(self.native)}"


class ScanParquetPlan(LazyPlan):
    """A Parquet file (a path, or its bytes), of which only `columns` are read."""

    __slots__ = ("columns", "source")

    def __init__(self, source: str | bytes, columns: Sequence[str]) -> None:
        self.source = source
        self.columns = tuple(columns)

    def _describe(self) -> str:
        source = self.source if isinstance(self.source, str) else "<bytes>"
        return f"SCAN PARQUET {source!r} {list(self.columns)}"


class OpPlan(LazyPlan):
    """A `DictDataFrame` method (or `group_by`/`concat`) applied to `inputs`."""

    __slots__ = ("args", "inputs", "kwargs", "name")

    def __init__(
        self,
        name: str,
        inputs: tuple[LazyPlan, ...],
        args: tuple[Any, ...] = (),
        kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.inputs = inputs
        self.args = args
        self.kwargs = dict(kwargs or {})

    def _describe(self) -> str:
        names = [arg for arg in self.args if isinstance(arg, str)]
        details = {
            key: value
            for key, value in self.kwargs.items()
            if isinstance(value, (str, int, bool, tuple, list)) or value is None
        }
        return " ".join(
            part
            for part in (self.name.upper(), str(names) if names else "", str(details))
            if part and part != "{}"
        )

    def with_inputs(self, inputs: tuple[LazyPlan, ...]) -> OpPlan:
        return OpPlan(self.name, inputs, self.args, self.kwargs)


# --- Execution -----------------------------------------------------------------------


Memo = dict[int, tuple[LazyPlan, DictDataFrame]]
"""Results by plan `id`. Each entry keeps its plan alive, so that `id` can't be reused."""

_DTYPE_INDEPENDENT_SELECTORS = frozenset({"selectors.all", "selectors.matches"})


class _DtypeDependentError(Exception):
    """A dry run can't tell this plan's output columns: they depend on dtypes."""


def _nodes_read_dtypes(nodes: Iterable[Any]) -> bool:
    for node in nodes:
        if node.kind.name == "SELECTOR" and node.name not in _DTYPE_INDEPENDENT_SELECTORS:
            return True
        for arg in node.exprs:
            if (sub_nodes := getattr(arg, "_nodes", None)) is not None and (
                _nodes_read_dtypes(sub_nodes)
            ):
                return True
    return False


def _reads_dtypes(plan: OpPlan) -> bool:
    """Whether `plan`'s output columns may depend on its input's dtypes.

    Dry runs infer every dtype as `Unknown`, so e.g. `with_columns(ncs.numeric())`
    would select nothing.
    """
    values: list[Any] = [*plan.args, *plan.kwargs.values()]
    while values:
        value = values.pop()
        if isinstance(value, (list, tuple)):
            values.extend(value)
            continue
        try:
            nodes = list(value._metadata.iter_nodes_reversed())
        except Exception:  # noqa: BLE001, S112
            continue
        if _nodes_read_dtypes(nodes):
            return True
    return False


def _execute(
    plan: LazyPlan, context: Context, memo: Memo, *, dry_run: bool = False
) -> DictDataFrame:
    """Run `plan`, reusing `memo`'s results for shared subplans.

    A `dry_run` runs it on empty frames, which is cheap and tells the output columns,
    and raises most errors (e.g. missing columns) that running it for real would.
    """
    if (entry := memo.get(id(plan))) is not None:
        return entry[1]
    if isinstance(plan, FramePlan):
        native: DictFrame = {name: [] for name in plan.native} if dry_run else plan.native
        result = DictDataFrame.from_native(native, context=context)
    elif isinstance(plan, ScanParquetPlan):
        result = DictDataFrame.from_native(
            {name: [] for name in plan.columns}  # type: ignore[arg-type]
            if dry_run
            else read_parquet(plan.source, plan.columns),
            context=context,
        )
    else:
        assert isinstance(plan, OpPlan)  # noqa: S101
        if dry_run and _reads_dtypes(plan):
            raise _DtypeDependentError
        frames = [
            _execute(child, context, memo, dry_run=dry_run) for child in plan.inputs
        ]
        result = _apply(plan, frames)
    memo[id(plan)] = (plan, result)
    return result


def _apply(plan: OpPlan, frames: list[DictDataFrame]) -> DictDataFrame:
    frame, *others = frames
    if plan.name in {"join", "join_asof"}:
        return getattr(frame, plan.name)(others[0], **plan.kwargs)
    if plan.name == "group_by":
        grouped = frame.group_by(
            plan.kwargs["keys"], drop_null_keys=plan.kwargs["drop_null_keys"]
        )
        return grouped.agg(*plan.kwargs["aggs"])
    if plan.name == "concat":
        namespace = frame.__narwhals_namespace__()
        return namespace.concat(frames, how=plan.kwargs["how"])
    return getattr(frame, plan.name)(*plan.args, **plan.kwargs)


# --- Projection pushdown -------------------------------------------------------------


def _nodes_columns(nodes: Iterable[Any]) -> set[str] | None:
    """The columns read by an expression with these (Narwhals) `nodes`, if known."""
    columns: set[str] = set()
    for node in nodes:
        if node.name == "col":
            columns.update(node.kwargs["names"])
        elif node.name in {"all", "exclude", "nth"} or node.kind.name == "SELECTOR":
            # These depend on which columns the input has (and in which order),
            # so pruning the input would change their meaning.
            return None
        for key in ("partition_by", "order_by"):
            columns.update(node.kwargs.get(key) or ())
        for arg in node.exprs:
            if isinstance(arg, str):
                if not node.str_as_lit:
                    columns.add(arg)
            elif (sub_nodes := getattr(arg, "_nodes", None)) is not None:
                if (sub_columns := _nodes_columns(sub_nodes)) is None:
                    return None
                columns |= sub_columns
    return columns


def _expr_columns(expr: Any) -> set[str] | None:
    if isinstance(expr, str):
        return {expr}
    try:
        nodes = list(expr._metadata.iter_nodes_reversed())
    except Exception:  # noqa: BLE001
        # e.g. built by Narwhals internally, without metadata: assume it reads anything.
        return None
    return _nodes_columns(nodes)


def _exprs_columns(exprs: Iterable[Any]) -> Columns:
    columns: set[str] = set()
    for expr in exprs:
        if (expr_columns := _expr_columns(expr)) is None:
            return None
        columns |= expr_columns
    return frozenset(columns)


def _union(*parts: Columns | Iterable[str]) -> Columns:
    if any(part is None for part in parts):
        return None
    return frozenset().union(*parts)  # type: ignore[arg-type]


def _requirements(  # noqa: C901, PLR0911, PLR0912
    plan: OpPlan, needed: Columns, input_frame: Callable[[LazyPlan], DictDataFrame]
) -> list[Columns]:
    """The columns each of `plan.inputs` must provide, for `plan` to provide `needed`."""
    name, args, kwargs = plan.name, plan.args, plan.kwargs
    if name in {"select", "aggregate"}:
        return [_exprs_columns(args)]
    if name == "simple_select":
        return [frozenset(args)]
    if name == "group_by":
        return [_exprs_columns((*kwargs["keys"], *kwargs["aggs"]))]
    if name == "join":
        return _join_requirements(plan, needed, input_frame)
    if needed is None:
        return [None] * len(plan.inputs)
    if name == "with_columns":
        outputs = {
            alias
            for expr in args
            for alias in expr._evaluate_aliases(input_frame(plan.inputs[0]))
        }
        return [_union(needed - outputs, _exprs_columns(args))]
    if name == "filter":
        predicate = args[0]
        if isinstance(predicate, list):  # a boolean mask
            return [needed]
        return [_union(needed, _exprs_columns(args))]
    if name == "sort":
        return [_union(needed, args)]
    if name == "concat":
        # Vertical and diagonal concatenation stack columns by name, so every input
        # needs to provide the same ones. If none are needed, keep them all, as each
        # input would otherwise keep a different one (see `_prune_leaf`).
        if kwargs["how"] == "horizontal" or not needed:
            return [None] * len(plan.inputs)
        return [needed] * len(plan.inputs)
    if name == "top_k":
        return [_union(needed, kwargs["by"])]
    if name in {"head", "tail"}:
        return [needed]
    if name == "drop":
        # Dropped columns aren't needed: `optimize` makes the drop non-strict.
        # If nothing is needed (e.g. only the row count), the column `_prune_leaf`
        # keeps to carry the length could be one that's dropped, so keep them all.
        return [needed - set(args[0]) if needed else None]
    if name in {"explode", "drop_nulls", "unique"}:
        subset = args[0]
        if not subset:  # e.g. `unique()` compares whole rows
            return [None]
        return [_union(needed, subset, kwargs.get("order_by") or ())]
    if name == "rename":
        mapping: Mapping[str, str] = args[0]
        inverse = {new: old for old, new in mapping.items()}
        return [
            frozenset(inverse.get(column, column) for column in needed) | set(mapping)
        ]
    if name == "cast":
        return [_union(needed, args[0])]
    if name == "with_row_index":
        index_name, order_by = args
        return [_union(needed - {index_name}, order_by or ())]
    return [None] * len(plan.inputs)


def _join_requirements(
    plan: OpPlan, needed: Columns, input_frame: Callable[[LazyPlan], DictDataFrame]
) -> list[Columns]:
    if needed is None:
        return [None, None]
    how: JoinStrategy = plan.kwargs["how"]
    left_on = plan.kwargs["left_on"] or ()
    right_on = plan.kwargs["right_on"] or ()
    if how in {"semi", "anti"}:
        return [_union(needed, left_on), frozenset(right_on)]
    left_columns = set(input_frame(plan.inputs[0]).columns)
    right_columns = input_frame(plan.inputs[1]).columns
    # Mirrors `DictDataFrame._join_output_names`: `inner`/`left` drop the right keys,
    # and right columns which collide with left ones get the suffix.
    excluded = set(right_on) if how in {"inner", "left"} else set()
    suffix = plan.kwargs["suffix"]
    right_kept = {
        column
        for column in right_columns
        if column not in excluded
        and (f"{column}{suffix}" if column in left_columns else column) in needed
    }
    # Keep a left column which a kept right one collides with, so the suffix still applies.
    left_needed = (needed & left_columns) | set(left_on) | (right_kept & left_columns)
    return [frozenset(left_needed), frozenset(right_kept | set(right_on))]


def _topological(plan: LazyPlan) -> list[LazyPlan]:
    """Every node of `plan` once, each before its inputs."""
    order: list[LazyPlan] = []
    seen: set[int] = set()

    def visit(node: LazyPlan) -> None:
        if id(node) in seen:
            return
        seen.add(id(node))
        for child in node.inputs:
            visit(child)
        order.append(node)

    visit(plan)
    return order[::-1]


def _prune_leaf(plan: LazyPlan, needed: Columns) -> LazyPlan:
    if needed is None:
        return plan
    if isinstance(plan, FramePlan):
        columns: Sequence[str] = list(plan.native)
    else:
        assert isinstance(plan, ScanParquetPlan)  # noqa: S101
        columns = plan.columns
    # Keep at least one column, or the number of rows would be lost.
    kept = [column for column in columns if column in needed] or list(columns[:1])
    if len(kept) == len(columns):
        return plan
    if isinstance(plan, FramePlan):
        return FramePlan({column: plan.native[column] for column in kept})
    return ScanParquetPlan(plan.source, kept)


def _rebuild(
    node: OpPlan,
    inputs: tuple[LazyPlan, ...],
    needed: Columns,
    input_frame: Callable[[LazyPlan], DictDataFrame],
) -> OpPlan:
    """`node`, on its pruned `inputs`."""
    if node.name == "drop":
        new_node = node.with_inputs(inputs)
        # Validated when it was defined: its columns may now be pruned away.
        new_node.kwargs["strict"] = False
        return new_node
    if node.name == "concat" and node.kwargs["how"] == "vertical":
        return _align_vertical_concat(node, inputs, needed, input_frame)
    return node.with_inputs(inputs)


def _align_vertical_concat(
    node: OpPlan,
    inputs: tuple[LazyPlan, ...],
    needed: Columns,
    input_frame: Callable[[LazyPlan], DictDataFrame],
) -> OpPlan:
    """Select the same columns, in the same order, from each pruned input.

    Pruning only narrows the leaves, so an input which computes its columns (e.g. a
    `select`) may still provide more of them than one which doesn't.
    """
    if not needed:
        return node
    try:
        output = input_frame(node).columns
    except Exception:  # noqa: BLE001
        return node
    columns = tuple(column for column in output if column in needed)
    aligned = tuple(OpPlan("simple_select", (child,), columns) for child in inputs)
    return node.with_inputs(aligned)


def optimize(plan: LazyPlan, context: Context, dry_runs: Memo | None = None) -> LazyPlan:
    """Projection pushdown: drop columns which nothing downstream reads."""
    dry_runs = {} if dry_runs is None else dry_runs

    def input_frame(node: LazyPlan) -> DictDataFrame:
        return _execute(node, context, dry_runs, dry_run=True)

    order = _topological(plan)
    # The result needs exactly its output columns, e.g. not those it drops. If a dry run
    # fails (some operations can't run on empty frames), keep every column instead.
    try:
        root_needs: Columns = frozenset(input_frame(plan).columns)
    except Exception:  # noqa: BLE001
        root_needs = None
    needs: dict[int, Columns] = {id(plan): root_needs}
    for node in order:
        if not isinstance(node, OpPlan):
            continue
        try:
            requirements = _requirements(node, needs[id(node)], input_frame)
        except Exception:  # noqa: BLE001
            requirements = [None] * len(node.inputs)
        for child, child_needs in zip(node.inputs, requirements, strict=True):
            # A plan shared by several operations must provide what each one needs.
            needs[id(child)] = (
                _union(needs[id(child)], child_needs)
                if id(child) in needs
                else child_needs
            )
    rebuilt: dict[int, LazyPlan] = {}
    for node in reversed(order):
        if isinstance(node, OpPlan):
            inputs = tuple(rebuilt[id(child)] for child in node.inputs)
            changed = any(
                new is not old for new, old in zip(inputs, node.inputs, strict=True)
            )
            rebuilt[id(node)] = (
                _rebuild(node, inputs, needs[id(node)], input_frame) if changed else node
            )
        else:
            rebuilt[id(node)] = _prune_leaf(node, needs[id(node)])
    return rebuilt[id(plan)]


# --- Lazy frames ---------------------------------------------------------------------


class DictLazyFrame(
    CompliantLazyFrame["DictExpr", LazyPlan, "LazyFrame[Any]"]  # type: ignore[type-var]
):
    _implementation = Implementation.UNKNOWN

    def __init__(
        self, plan: LazyPlan, *, version: Version, dry_runs: Memo | None = None
    ) -> None:
        self._native_frame = plan
        self._version = version
        # Shared with the frames this one is built from, which ran its subplans already.
        self._dry_runs: Memo = {} if dry_runs is None else dry_runs
        self._cached_columns: list[str] | None = None
        self._cached_schema: dict[str, DType] | None = None

    # Protocol plumbing
    def __narwhals_lazyframe__(self) -> Self:
        return self

    def __narwhals_namespace__(self) -> DictNamespace:
        from pizzlies._plugin.namespace import DictNamespace

        return DictNamespace(version=self._version)

    def __native_namespace__(self) -> ModuleType:
        import pizzlies

        return pizzlies

    @classmethod
    def from_native(cls, data: LazyPlan, /, *, context: Context) -> Self:
        return cls(data, version=context._version)

    @staticmethod
    def _is_native(obj: Any) -> TypeIs[LazyPlan]:
        return isinstance(obj, LazyPlan)

    def _with_native(self, df: LazyPlan) -> Self:
        return self.__class__(df, version=self._version)

    def _with_version(self, version: Version) -> Self:
        return self.__class__(self.native, version=version)

    def to_narwhals(self) -> LazyFrame[Any]:
        return self._version.lazyframe(self, level="lazy")

    def native_repr(self) -> str:
        return fit_lines(self.native.explain())

    def _op(
        self, name: str, *args: Any, others: Sequence[DictLazyFrame] = (), **kwargs: Any
    ) -> Self:
        plan = OpPlan(
            name, (self.native, *(other.native for other in others)), args, kwargs
        )
        dry_runs = self._dry_runs
        for other in others:
            dry_runs = {**dry_runs, **other._dry_runs}
        result = self.__class__(plan, version=self._version, dry_runs=dry_runs)
        # Raise invalid operations (e.g. on missing columns) now, like eager frames do.
        result.columns  # noqa: B018
        return result

    # Execution
    def _collect_frame(self) -> DictDataFrame:
        return _execute(optimize(self.native, self, self._dry_runs), self, {})

    def collect(self, backend: Implementation | None, **kwargs: Any) -> Any:
        return self._collect_frame().collect(backend, **kwargs)

    @property
    def columns(self) -> list[str]:
        if self._cached_columns is None:
            try:
                frame = _execute(self.native, self, self._dry_runs, dry_run=True)
            except Exception:  # noqa: BLE001
                # Some operations can't run on empty frames: run them for real, and
                # unoptimized, so that any error is exactly the eager one.
                frame = _execute(self.native, self, {})
            self._cached_columns = list(frame.columns)
        return self._cached_columns

    @property
    def schema(self) -> dict[str, DType]:
        # Dtypes are inferred from the values, so the plan has to run.
        if self._cached_schema is None:
            self._cached_schema = dict(self._collect_frame().schema)
        return self._cached_schema

    def collect_schema(self) -> dict[str, DType]:
        return self.schema

    def _iter_columns(self) -> Iterator[DictSeries]:
        for name, values in self._collect_frame().native.items():
            yield DictSeries(values, name=name, version=self._version)

    sink_parquet: Any = not_implemented()

    # Operations, recorded in the plan
    def select(self, *exprs: DictExpr) -> Self:
        return self._op("select", *exprs)

    def aggregate(self, *exprs: DictExpr) -> Self:
        return self._op("aggregate", *exprs)

    def simple_select(self, *column_names: str) -> Self:
        return self._op("simple_select", *column_names)

    def with_columns(self, *exprs: DictExpr) -> Self:
        return self._op("with_columns", *exprs)

    def filter(self, predicate: DictExpr | Any) -> Self:
        return self._op("filter", predicate)

    def drop(self, columns: Sequence[str], *, strict: bool) -> Self:
        return self._op("drop", list(columns), strict=strict)

    def drop_nulls(self, subset: Sequence[str] | None) -> Self:
        return self._op("drop_nulls", subset)

    def explode(
        self,
        columns: Sequence[str],
        *,
        empty_as_null: bool = True,
        keep_nulls: bool = True,
    ) -> Self:
        return self._op(
            "explode", list(columns), empty_as_null=empty_as_null, keep_nulls=keep_nulls
        )

    def head(self, n: int) -> Self:
        return self._op("head", n)

    def top_k(self, k: int, *, by: Iterable[str], reverse: bool | Sequence[bool]) -> Self:
        return self._op("top_k", k, by=list(by), reverse=reverse)

    def rename(self, mapping: Mapping[str, str]) -> Self:
        return self._op("rename", dict(mapping))

    def cast(self, dtypes: Mapping[str, IntoDType]) -> Self:
        return self._op("cast", dict(dtypes))

    def sort(self, *by: str, descending: bool | Sequence[bool], nulls_last: bool) -> Self:
        return self._op("sort", *by, descending=descending, nulls_last=nulls_last)

    def unique(
        self,
        subset: Sequence[str] | None,
        *,
        keep: Any,
        maintain_order: bool | None = None,
        order_by: Sequence[str] | None = None,
    ) -> Self:
        return self._op(
            "unique", subset, keep=keep, maintain_order=maintain_order, order_by=order_by
        )

    def unpivot(
        self,
        on: Sequence[str] | None,
        index: Sequence[str] | None,
        variable_name: str,
        value_name: str,
    ) -> Self:
        return self._op("unpivot", on, index, variable_name, value_name)

    def with_row_index(self, name: str, order_by: Sequence[str] | None) -> Self:
        return self._op("with_row_index", name, order_by)

    def join(
        self,
        other: Self,
        *,
        how: JoinStrategy,
        left_on: Sequence[str] | None,
        right_on: Sequence[str] | None,
        suffix: str,
    ) -> Self:
        return self._op(
            "join",
            others=(other,),
            how=how,
            left_on=left_on,
            right_on=right_on,
            suffix=suffix,
        )

    def join_asof(
        self,
        other: Self,
        *,
        left_on: str,
        right_on: str,
        by_left: Sequence[str] | None,
        by_right: Sequence[str] | None,
        strategy: AsofJoinStrategy,
        suffix: str,
    ) -> Self:
        return self._op(
            "join_asof",
            others=(other,),
            left_on=left_on,
            right_on=right_on,
            by_left=by_left,
            by_right=by_right,
            strategy=strategy,
            suffix=suffix,
        )

    def group_by(
        self, keys: Sequence[str] | Sequence[DictExpr], *, drop_null_keys: bool
    ) -> DictLazyGroupBy:
        return DictLazyGroupBy(self, keys, drop_null_keys=drop_null_keys)


class DictLazyGroupBy(CompliantGroupBy[DictLazyFrame, "DictExpr"]):
    """Records `group_by(...).agg(...)` as a single plan operation."""

    def __init__(
        self,
        compliant_frame: DictLazyFrame,
        keys: Sequence[DictExpr] | Sequence[str],
        /,
        *,
        drop_null_keys: bool,
    ) -> None:
        self._compliant_frame = compliant_frame
        self._keys = list(keys)
        self._drop_null_keys = drop_null_keys

    def agg(self, *exprs: DictExpr) -> DictLazyFrame:
        return self.compliant._op(
            "group_by", keys=self._keys, drop_null_keys=self._drop_null_keys, aggs=exprs
        )


def lazy_concat(items: Sequence[DictLazyFrame], *, how: ConcatMethod) -> DictLazyFrame:
    first, *rest = items
    return first._op("concat", others=rest, how=how)
