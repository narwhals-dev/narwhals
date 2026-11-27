"""Lazy frames for `pizzlies`, with projection pushdown.

A `DictLazyFrame` doesn't compute anything: each method records an operation in a
`LazyPlan`, whose leaves are in-memory frames or Parquet scans. `collect` first works
out each operation's output column names, which raises name errors (e.g. missing or
duplicate columns) as eager mode would. It then drops the leaf columns which nothing
reads, so that scans only decode the columns a query uses, and replays the plan with
`DictDataFrame`.

Output names are worked out without running anything, except for dtype selectors:
dtypes are inferred from values. Whenever it's unclear which columns an expression
reads (`all()`, selectors, ...), its input keeps every column.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from narwhals.compliant import CompliantGroupBy, CompliantLazyFrame
from narwhals.exceptions import ColumnNotFoundError, ShapeError
from narwhals.utils import Implementation
from pizzlies._plugin._vendored import (
    check_column_names_are_unique,
    evaluate_output_names_and_aliases,
    not_implemented,
)
from pizzlies._plugin.dataframe import DictDataFrame
from pizzlies._plugin.parquet import read_parquet
from pizzlies._plugin.repr import fit_lines
from pizzlies._plugin.series import DictSeries
from pizzlies._plugin.utils import asof_excluded_right, exprs_input_columns

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
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
    """Columns which a plan must provide, or `None` for all of them."""


# --- Plans ---------------------------------------------------------------------------


class LazyPlan:
    """A query plan: the native object of `pizzlies`' lazy frames. Immutable."""

    __slots__ = ("schema",)
    inputs: tuple[LazyPlan, ...] = ()
    # The output dtypes, once a `DictLazyFrame` has run the plan to infer them.
    schema: dict[str, DType] | None

    def explain(self) -> str:
        """The plan as an indented tree, operations above their inputs.

        A subplan used more than once is shown once, and then referred to by number.
        """
        uses: dict[int, int] = {}
        for node in _topological(self):
            for child in node.inputs:
                uses[id(child)] = uses.get(id(child), 0) + 1
        labels: dict[int, int] = {}
        lines: list[str] = []
        stack: list[tuple[LazyPlan, int]] = [(self, 0)]
        while stack:
            node, depth = stack.pop()
            indent = "  " * depth
            if (label := labels.get(id(node))) is not None:
                lines.append(f"{indent}<see [{label}]>")
                continue
            prefix = ""
            if uses.get(id(node), 0) > 1:
                labels[id(node)] = label = len(labels) + 1
                prefix = f"[{label}] "
            lines.append(f"{indent}{prefix}{node._describe()}")
            stack.extend((child, depth + 1) for child in reversed(node.inputs))
        return "\n".join(lines)

    def _describe(self) -> str:  # pragma: no cover
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"LazyPlan:\n{self.explain()}"


class FramePlan(LazyPlan):
    """An in-memory dict of lists."""

    __slots__ = ("native",)

    def __init__(self, native: DictFrame) -> None:
        # A shallow copy, so that adding or replacing columns in `native` later doesn't
        # change the plan. The column lists themselves are shared, not copied.
        self.native = dict(native)
        self.schema = None

    def _describe(self) -> str:
        return f"FRAME {list(self.native)}"


class ScanParquetPlan(LazyPlan):
    """A Parquet file (a path, or its bytes), of which only `columns` are read."""

    __slots__ = ("columns", "source")

    def __init__(self, source: str | bytes, columns: Sequence[str]) -> None:
        self.source = source
        self.columns = tuple(columns)
        self.schema = None

    def _describe(self) -> str:
        source = self.source if isinstance(self.source, str) else "<bytes>"
        return f"SCAN PARQUET {source!r} {list(self.columns)}"


class OpPlan(LazyPlan):
    """A `DictDataFrame` method (or `group_by`/`concat`) applied to `inputs`."""

    __slots__ = ("args", "inputs", "kwargs", "name", "names")

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
        self.names: list[str] | None = None  # output names, see `_output_names`
        self.schema = None

    def _describe(self) -> str:
        names = [arg for arg in self.args if isinstance(arg, (str, list, tuple, dict))]
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


def _execute(plan: LazyPlan, context: Context) -> DictDataFrame:
    # Inputs first. `order` keeps every node alive, so that their `id`s can't be reused.
    order = _topological(plan)[::-1]
    uses: dict[int, int] = {}
    for node in order:
        for child in node.inputs:
            uses[id(child)] = uses.get(id(child), 0) + 1
    # A subplan used more than once (e.g. in a self-join) runs once, and each result
    # is dropped once its last user has run.
    results: dict[int, DictDataFrame] = {}
    for node in order:
        if isinstance(node, FramePlan):
            result = DictDataFrame.from_native(node.native, context=context)
        elif isinstance(node, ScanParquetPlan):
            native: Any = read_parquet(node.source, node.columns)
            result = DictDataFrame.from_native(native, context=context)
        else:
            assert isinstance(node, OpPlan)  # noqa: S101
            result = _apply(node, [results[id(child)] for child in node.inputs])
            for child in node.inputs:
                uses[id(child)] -= 1
                if not uses[id(child)]:
                    del results[id(child)]
        results[id(node)] = result
    return results[id(plan)]


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


# --- Column names --------------------------------------------------------------------


def _check_exist(subset: Iterable[str], available: Sequence[str]) -> None:
    if missing := set(subset).difference(available):
        raise ColumnNotFoundError.from_missing_and_available_column_names(
            missing, available
        )


def _check_reads(exprs: Iterable[Any], available: Sequence[str]) -> None:
    if (columns := exprs_input_columns(exprs)) is not None:
        _check_exist(columns, available)


def _output_names(plan: LazyPlan, version: Version) -> list[str]:
    """The column names `plan` produces, without running it (except dtype selectors).

    Also raises the errors which only depend on names, e.g. missing columns.
    """
    if isinstance(plan, FramePlan):
        return list(plan.native)
    if isinstance(plan, ScanParquetPlan):
        return list(plan.columns)
    assert isinstance(plan, OpPlan)  # noqa: S101
    if plan.names is None:
        # Inputs first, in a loop rather than recursively: plans can be deep.
        for node in reversed(_topological(plan)):
            if isinstance(node, OpPlan) and node.names is None:
                node.names = _op_output_names(node, version)
    assert plan.names is not None  # noqa: S101
    return list(plan.names)


def _aliases(
    exprs: Iterable[Any], frame: DictLazyFrame, exclude: Sequence[str] = ()
) -> list[str]:
    # Evaluated against the lazy input: only dtype selectors need its schema, i.e. to run it.
    return [
        alias
        for expr in exprs
        for alias in evaluate_output_names_and_aliases(expr, frame, exclude)[1]
    ]


def _join_names(
    left: Sequence[str],
    right: Sequence[str],
    *,
    exclude_right: Collection[str],
    suffix: str,
) -> dict[str, str]:
    """Right column names to output names, as `DictDataFrame._join_output_names`."""
    left_set = set(left)
    right_names = {
        name: f"{name}{suffix}" if name in left_set else name
        for name in right
        if name not in exclude_right
    }
    check_column_names_are_unique([*left, *right_names.values()])
    return right_names


def _group_by_names(
    keys: Sequence[Any], aggs: Sequence[Any], frame: DictLazyFrame
) -> list[str]:
    names = frame.columns
    _check_reads(keys, names)
    _check_reads(aggs, names)
    key_names = [key for key in keys if isinstance(key, str)]
    key_exprs = [key for key in keys if not isinstance(key, str)]
    output_keys = [*key_names, *_aliases(key_exprs, frame)]
    result = [*output_keys, *_aliases(aggs, frame, exclude=[*key_names, *output_keys])]
    check_column_names_are_unique(result)
    return result


def _join_output(
    name: str, left: list[str], right: list[str], kwargs: Mapping[str, Any]
) -> list[str]:
    if name == "join_asof":
        _check_exist([kwargs["left_on"], *(kwargs["by_left"] or ())], left)
        _check_exist([kwargs["right_on"], *(kwargs["by_right"] or ())], right)
        exclude = asof_excluded_right(
            kwargs["left_on"], kwargs["right_on"], kwargs["by_right"]
        )
    else:
        _check_exist(kwargs["left_on"] or (), left)
        _check_exist(kwargs["right_on"] or (), right)
        if kwargs["how"] in {"semi", "anti"}:
            return left
        exclude = set(kwargs["right_on"]) if kwargs["how"] in {"inner", "left"} else set()
    right_names = _join_names(left, right, exclude_right=exclude, suffix=kwargs["suffix"])
    return [*left, *right_names.values()]


def _concat_names(all_names: list[list[str]], how: ConcatMethod) -> list[str]:
    if how == "horizontal":
        result = [column for names in all_names for column in names]
        check_column_names_are_unique(result)
        return result
    if how == "vertical":
        for i, names in enumerate(all_names[1:], start=1):
            if names != all_names[0]:
                msg = (
                    "unable to vstack, column names don't match:\n"
                    f"   - dataframe 0: {all_names[0]}\n"
                    f"   - dataframe {i}: {names}\n"
                )
                raise ShapeError(msg)
        return all_names[0]
    return list(dict.fromkeys(column for names in all_names for column in names))


def _op_output_names(plan: OpPlan, version: Version) -> list[str]:  # noqa: C901, PLR0911, PLR0912
    name, args, kwargs = plan.name, plan.args, plan.kwargs
    frame = DictLazyFrame(plan.inputs[0], version=version)
    names = frame.columns
    if name in {"select", "aggregate"}:
        _check_reads(args, names)
        result = _aliases(args, frame)
        check_column_names_are_unique(result)
        return result
    if name == "simple_select":
        _check_exist(args, names)
        check_column_names_are_unique(args)
        return list(args)
    if name == "with_columns":
        _check_reads(args, names)
        check_column_names_are_unique(new := _aliases(args, frame))
        return list(dict.fromkeys([*names, *new]))
    if name == "filter":
        _check_reads(args, names)
        return names
    if name == "sort":
        _check_exist(args, names)
        return names
    if name == "top_k":
        _check_exist(kwargs["by"], names)
        return names
    if name in {"drop_nulls", "unique"}:
        _check_exist([*(args[0] or ()), *(kwargs.get("order_by") or ())], names)
        return names
    if name in {"head", "explode", "cast"}:
        _check_exist(args[0] if name != "head" else (), names)
        return names
    if name == "drop":
        if kwargs["strict"]:
            _check_exist(args[0], names)
        return [column for column in names if column not in set(args[0])]
    if name == "rename":
        mapping: Mapping[str, str] = args[0]
        _check_exist(mapping, names)
        result = [mapping.get(column, column) for column in names]
        check_column_names_are_unique(result)
        return result
    if name == "with_row_index":
        _check_exist(args[1] or (), names)
        result = [args[0], *names]
        check_column_names_are_unique(result)
        return result
    if name == "unpivot":
        on, index, variable_name, value_name = args
        index = list(index or ())
        on = [column for column in names if column not in index] if on is None else on
        _check_exist([*index, *on], names)
        result = [*index, variable_name, value_name]
        check_column_names_are_unique(result)
        return result
    if name == "group_by":
        return _group_by_names(kwargs["keys"], kwargs["aggs"], frame)
    if name in {"join", "join_asof"}:
        right = DictLazyFrame(plan.inputs[1], version=version).columns
        return _join_output(name, names, right, kwargs)
    if name == "concat":
        all_names = [_output_names(child, version) for child in plan.inputs]
        return _concat_names(all_names, kwargs["how"])
    msg = f"Unknown plan operation: {name!r}."  # pragma: no cover
    raise AssertionError(msg)  # pragma: no cover


# --- Projection pushdown -------------------------------------------------------------


def _union(*parts: Columns | Iterable[str] | None) -> Columns:
    if any(part is None for part in parts):
        return None
    return frozenset().union(*parts)  # type: ignore[arg-type]


def _requirements(plan: OpPlan, needed: Columns, version: Version) -> list[Columns]:  # noqa: C901, PLR0911, PLR0912
    """The columns each of `plan.inputs` must provide, for `plan` to provide `needed`."""
    name, args, kwargs = plan.name, plan.args, plan.kwargs
    if name in {"select", "aggregate"}:
        return [exprs_input_columns(args)]
    if name == "simple_select":
        return [frozenset(args)]
    if name == "group_by":
        return [exprs_input_columns((*kwargs["keys"], *kwargs["aggs"]))]
    if (
        needed is None
        or (name == "unpivot" and args[0] is None)
        or (name == "concat" and kwargs["how"] != "vertical")
    ):
        return [None] * len(plan.inputs)
    if name == "unpivot":
        return [frozenset([*args[0], *(args[1] or ())])]
    input_names = _output_names(plan.inputs[0], version)
    if name == "with_columns":
        # Overwritten columns stay needed (if they are), so they keep their position.
        created = set(_output_names(plan, version)).difference(input_names)
        return [_union(needed - created, exprs_input_columns(args))]
    if name == "filter":
        return [_union(needed, exprs_input_columns(args))]
    if name in {"sort", "top_k"}:
        return [_union(needed, args if name == "sort" else kwargs["by"])]
    if name in {"head", "drop", "concat"}:
        # `drop` is made non-strict when pruned: it was checked against all columns.
        return [needed] * len(plan.inputs)
    if name in {"explode", "cast"}:
        return [_union(needed, args[0])]
    if name == "rename":
        mapping: Mapping[str, str] = args[0]
        inverse = {new: old for old, new in mapping.items()}
        return [
            frozenset({inverse.get(column, column) for column in needed} | set(mapping))
        ]
    if name == "with_row_index":
        return [_union(needed - {args[0]}, args[1] or ())]
    if name in {"drop_nulls", "unique"}:
        if not (subset := args[0]):  # e.g. `unique()` compares whole rows
            return [None]
        return [_union(needed, subset, kwargs.get("order_by") or ())]
    if name == "join" and kwargs["how"] in {"semi", "anti"}:
        return [_union(needed, kwargs["left_on"]), frozenset(kwargs["right_on"])]
    if name in {"join", "join_asof"}:
        right_names = _output_names(plan.inputs[1], version)
        if name == "join":
            left_on, right_on = kwargs["left_on"] or (), kwargs["right_on"] or ()
            exclude = set(right_on) if kwargs["how"] in {"inner", "left"} else set()
        else:
            left_on = [kwargs["left_on"], *(kwargs["by_left"] or ())]
            right_on = [kwargs["right_on"], *(kwargs["by_right"] or ())]
            exclude = asof_excluded_right(
                kwargs["left_on"], kwargs["right_on"], kwargs["by_right"]
            )
        output = _join_names(
            input_names, right_names, exclude_right=exclude, suffix=kwargs["suffix"]
        )
        right_kept = {column for column, out in output.items() if out in needed}
        # A kept right column keeps its left namesake, so that it's still suffixed.
        left_kept = (
            (needed & set(input_names)) | set(left_on) | (right_kept & set(input_names))
        )
        return [frozenset(left_kept), frozenset(right_kept | set(right_on))]
    return [None] * len(plan.inputs)


def _topological(plan: LazyPlan) -> list[LazyPlan]:
    """Every node of `plan` once, each before its inputs."""
    order: list[LazyPlan] = []
    seen: set[int] = set()
    # A depth-first post-order walk, in a loop: plans can be too deep to recurse.
    stack: list[tuple[LazyPlan, bool]] = [(plan, False)]
    while stack:
        node, inputs_done = stack.pop()
        if inputs_done:
            order.append(node)
            continue
        if id(node) in seen:
            continue
        seen.add(id(node))
        stack.append((node, True))
        stack.extend((child, False) for child in reversed(node.inputs))
    return order[::-1]


def _prune_leaf(plan: LazyPlan, needed: Columns) -> LazyPlan:
    if needed is None:
        return plan
    if isinstance(plan, FramePlan):
        kept = [column for column in plan.native if column in needed]
        if len(kept) == len(plan.native):
            return plan
        return FramePlan({column: plan.native[column] for column in kept})
    assert isinstance(plan, ScanParquetPlan)  # noqa: S101
    kept = [column for column in plan.columns if column in needed]
    return plan if len(kept) == len(plan.columns) else ScanParquetPlan(plan.source, kept)


def _rebuild(
    node: OpPlan, inputs: tuple[LazyPlan, ...], needed: Columns, version: Version
) -> OpPlan:
    new = node.with_inputs(inputs)
    if node.name == "drop":
        new.kwargs["strict"] = False
    elif (
        node.name == "concat" and node.kwargs["how"] == "vertical" and needed is not None
    ):
        # Pruned inputs which compute their columns may still have more than others.
        columns = tuple(c for c in _output_names(node, version) if c in needed)
        new.inputs = tuple(OpPlan("simple_select", (child,), columns) for child in inputs)
    return new


def optimize(plan: LazyPlan, version: Version) -> LazyPlan:
    """Projection pushdown: drop leaf columns which nothing reads."""
    order = _topological(plan)
    needs: dict[int, Columns] = {id(plan): frozenset(_output_names(plan, version))}
    for node in order:
        needed = needs[id(node)]
        if needed is not None and not needed and (names := _output_names(node, version)):
            # Keep one column, or the number of rows would be lost.
            needs[id(node)] = needed = frozenset(names[:1])
        if not isinstance(node, OpPlan):
            continue
        for child, child_needs in zip(
            node.inputs, _requirements(node, needed, version), strict=True
        ):
            # A subplan shared by several operations must provide what each one needs.
            needs[id(child)] = (
                _union(needs[id(child)], child_needs)
                if id(child) in needs
                else child_needs
            )
    rebuilt: dict[int, LazyPlan] = {}
    for node in reversed(order):
        if isinstance(node, OpPlan):
            inputs = tuple(rebuilt[id(child)] for child in node.inputs)
            unchanged = all(
                new is old for new, old in zip(inputs, node.inputs, strict=True)
            )
            rebuilt[id(node)] = (
                node if unchanged else _rebuild(node, inputs, needs[id(node)], version)
            )
        else:
            rebuilt[id(node)] = _prune_leaf(node, needs[id(node)])
    return rebuilt[id(plan)]


# --- Lazy frames ---------------------------------------------------------------------


class DictLazyFrame(
    CompliantLazyFrame["DictExpr", LazyPlan, "LazyFrame[Any]"]  # type: ignore[type-var]
):
    _implementation = Implementation.UNKNOWN

    def __init__(self, plan: LazyPlan, *, version: Version) -> None:
        self._native_frame = plan
        self._version = version

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
        inputs = (self.native, *(other.native for other in others))
        return self._with_native(OpPlan(name, inputs, args, kwargs))

    # Execution
    def _collect_frame(self) -> DictDataFrame:
        names = self.columns  # also raises e.g. missing column errors, before any work
        result = _execute(optimize(self.native, self._version), self)
        if result.columns != names:  # pragma: no cover
            msg = (
                f"pizzlies bug: expected the columns {names}, got {result.columns}. "
                "Please report it at https://github.com/narwhals-dev/narwhals/issues."
            )
            raise AssertionError(msg)
        return result

    def collect(self, backend: Implementation | None, **kwargs: Any) -> Any:
        return self._collect_frame().collect(backend, **kwargs)

    @property
    def columns(self) -> list[str]:
        return _output_names(self.native, self._version)

    @property
    def schema(self) -> dict[str, DType]:
        # Dtypes are inferred from values, so this runs the plan: once, as plans
        # are immutable.
        if (schema := self.native.schema) is None:
            schema = self.native.schema = dict(self._collect_frame().schema)
        return dict(schema)

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
