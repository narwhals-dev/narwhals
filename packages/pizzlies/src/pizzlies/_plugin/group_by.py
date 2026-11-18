from __future__ import annotations

from functools import partial
from itertools import chain
from operator import is_not, itemgetter
from typing import TYPE_CHECKING, Any, ClassVar

from narwhals.compliant import CompliantGroupBy
from pizzlies._plugin._vendored import (
    evaluate_aliases,
    evaluate_output_names_and_aliases,
    is_sequence_of,
)
from pizzlies._plugin.series import DictSeries

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence

    from pizzlies._plugin._vendored import NarwhalsAggregation
    from pizzlies._plugin.dataframe import DictDataFrame
    from pizzlies._plugin.expr import DictExpr
    from pizzlies._plugin.typing import DictFrame


def _bucket_by_keys(key_columns: Sequence[Sequence[Any]]) -> dict[Any, list[int]]:
    """Map each partition key to its row indices, in first-appearance order.

    A single key uses the raw column value directly rather than a tuple, multiple
    keys use the value tuple.
    """
    groups: dict[Any, list[int]] = {}
    it = (
        enumerate(key_columns[0])
        if len(key_columns) == 1
        else enumerate(zip(*key_columns, strict=True))
    )
    get = groups.get
    for index, key in it:
        if (rows := get(key)) is None:
            groups[key] = [index]
        else:
            rows.append(index)
    return groups


def _mean(values: list[Any]) -> Any:
    return sum(values) / len(values)


# Aggregations computed with builtins straight on each group's values, matching the
# `DictSeries` methods of the same name. Groups are never empty.
_NULL_SAFE_AGGS: Mapping[str, Callable[[list[Any]], Any]] = {
    "len": len,
    "first": itemgetter(0),
    "last": itemgetter(-1),
    "n_unique": lambda values: len(set(values)),
}
# These ones skip nulls, so only apply as-is to columns without any.
_NULL_FREE_AGGS: Mapping[str, Callable[[list[Any]], Any]] = {
    "sum": sum,
    "mean": _mean,
    "min": min,
    "max": max,
    "count": len,
    "any": any,
    "all": all,
}


_is_not_null = partial(is_not, None)


def _non_null(values: list[Any]) -> list[Any]:
    return list(filter(_is_not_null, values))


def _sum_skipping_nulls(values: list[Any]) -> Any:
    return sum(filter(_is_not_null, values))


def _mean_skipping_nulls(values: list[Any]) -> Any:
    return _mean(present) if (present := _non_null(values)) else None


def _min_skipping_nulls(values: list[Any]) -> Any:
    return min(present) if (present := _non_null(values)) else None


def _max_skipping_nulls(values: list[Any]) -> Any:
    return max(present) if (present := _non_null(values)) else None


# The same aggregations, for columns with nulls: filtering them out still runs in C.
_NULL_SKIPPING_AGGS: Mapping[str, Callable[[list[Any]], Any]] = {
    "sum": _sum_skipping_nulls,
    "mean": _mean_skipping_nulls,
    "min": _min_skipping_nulls,
    "max": _max_skipping_nulls,
    "count": lambda values: len(values) - values.count(None),
    "any": lambda values: any(filter(_is_not_null, values)),
    "all": lambda values: all(filter(_is_not_null, values)),
}


class _GroupedRows:
    """Per-`agg` cache of grouped row indices, gathered columns, and sub-frames.

    Everything is keyed by `order_by`, because `first`/`last` may each request a
    different one. Callers get the cached lists rather than copies, which is safe
    because series and frame operations never mutate native lists in place.
    """

    def __init__(self, frame: DictDataFrame, keys: Sequence[str]) -> None:
        self._frame = frame
        self._key_columns = [frame.native[key] for key in keys]
        # Keys are raw values for a single key, value tuples for several -- see
        # `_bucket_by_keys`. `row_keys` uses the matching representation so the
        # `order_by` re-bucketing below looks them up consistently.
        base_groups = _bucket_by_keys(self._key_columns)
        self.keys: list[Any] = list(base_groups.keys())
        self._indices: dict[tuple[str, ...], list[list[int]]] = {
            (): [base_groups[key] for key in self.keys]
        }

        # Row keys are only needed to re-bucket a non-trivial `order_by`, so they are built lazily
        self._row_keys: Sequence[Any] | None = None
        self._columns: dict[tuple[tuple[str, ...], str], list[list[Any]]] = {}
        self._frames: dict[
            tuple[tuple[str, ...], tuple[str, ...]], list[DictDataFrame]
        ] = {}

    @property
    def row_keys(self) -> Sequence[tuple[Any, ...]]:
        if self._row_keys is None:
            self._row_keys = (
                self._key_columns[0]
                if len(self._key_columns) == 1
                else list(zip(*self._key_columns, strict=True))
            )
        return self._row_keys

    def key_column(self, position: int) -> list[Any]:
        """The distinct key values for key-column `position`, in group order."""
        if len(self._key_columns) == 1:
            return self.keys
        return [key[position] for key in self.keys]

    def indices(self, order_by: tuple[str, ...]) -> list[list[int]]:
        """Each group's row indices, ordered by `order_by` (ascending, nulls first)."""
        if (cached := self._indices.get(order_by)) is None:
            buckets: dict[Any, list[int]] = {key: [] for key in self.keys}
            for index in self._frame._sorted_indices(
                order_by, descending=False, nulls_last=False
            ):
                buckets[self.row_keys[index]].append(index)
            cached = self._indices[order_by] = [buckets[key] for key in self.keys]
        return cached

    def gather(self, order_by: tuple[str, ...], name: str) -> list[list[Any]]:
        """Column `name` split into one list of values per group, in `order_by` order."""
        if (columns := self._columns.get(cache_key := (order_by, name))) is None:
            columns = self._columns[cache_key] = self.split(
                self._frame.native[name], order_by
            )
        return columns

    def split(self, values: Sequence[Any], order_by: tuple[str, ...]) -> list[list[Any]]:
        """`values`, one per row, split into one list per group in `order_by` order."""
        get = values.__getitem__
        return [list(map(get, indices)) for indices in self.indices(order_by)]

    def frames(
        self, order_by: tuple[str, ...], columns: tuple[str, ...]
    ) -> list[DictDataFrame]:
        """One sub-frame per group with `columns`, rows in `order_by` order."""
        if (frames := self._frames.get(cache_key := (order_by, columns))) is None:
            frames = self._frames[cache_key] = [
                self._frame._with_native(
                    {name: self.gather(order_by, name)[position] for name in columns},
                    validate_column_names=False,
                )
                for position in range(len(self.keys))
            ]
        return frames


class DictGroupBy(CompliantGroupBy["DictDataFrame", "DictExpr"]):
    """Group-by for the dict backend.

    The "native" aggregation is simply the name of the `DictSeries` method to
    call: each group gathers its rows into a `DictSeries` and delegates.
    """

    _REMAP_AGGS: ClassVar[Mapping[NarwhalsAggregation, str]] = {
        "sum": "sum",
        "mean": "mean",
        "median": "median",
        "max": "max",
        "min": "min",
        "std": "std",
        "var": "var",
        "len": "len",
        "n_unique": "n_unique",
        "count": "count",
        "all": "all",
        "any": "any",
        "first": "first",
        "last": "last",
        "any_value": "any_value",
    }
    _ORDER_DEPENDENT: ClassVar[frozenset[NarwhalsAggregation]] = frozenset(
        ("first", "last", "any_value")
    )

    def __init__(
        self,
        df: DictDataFrame,
        keys: Sequence[DictExpr] | Sequence[str],
        /,
        *,
        drop_null_keys: bool,
    ) -> None:
        self._df = df
        frame, self._keys, self._output_key_names = self._parse_keys(df, keys=keys)
        self._compliant_frame = frame.drop_nulls(self._keys) if drop_null_keys else frame

    def _parse_keys(
        self, frame: DictDataFrame, keys: Sequence[DictExpr] | Sequence[str]
    ) -> tuple[DictDataFrame, list[str], list[str]]:
        if is_sequence_of(keys, str):
            keys_str = list(keys)
            return frame, keys_str, keys_str.copy()
        return self._parse_expr_keys(frame, keys=keys)

    @staticmethod
    def _parse_expr_keys(
        frame: DictDataFrame, keys: Sequence[DictExpr]
    ) -> tuple[DictDataFrame, list[str], list[str]]:
        """Evaluate key expressions into temporary columns which don't clash with existing ones.

        The output key names are returned too, so the temporary columns can be renamed
        back after aggregating, and excluded from unnamed expressions in `.agg(...)`.
        """
        tmp_name_length = max(len(str(c)) for c in frame.columns) + 1

        def _temporary_name(key: str) -> str:
            # 5 is the length of `__tmp`
            return f"_{key}_tmp{'_' * (tmp_name_length - len(key) - 5)}"

        keys_aliases = [expr._evaluate_aliases(frame) for expr in keys]
        safe_keys = [
            # multi-output expression cannot have duplicate names, hence it's safe to suffix
            key.name.map(_temporary_name)
            if (metadata := key._metadata) and metadata.expansion_kind.is_multi_output()
            # otherwise it's single named and we can use Expr.alias
            else key.alias(_temporary_name(new_names[0]))
            for key, new_names in zip(keys, keys_aliases, strict=True)
        ]
        return (
            frame.with_columns(*safe_keys),
            evaluate_aliases(frame, safe_keys),
            list(chain.from_iterable(keys_aliases)),
        )

    @classmethod
    def _is_simple(cls, expr: DictExpr, /) -> bool:
        return expr._is_elementary() and cls._leaf_name(expr) in cls._REMAP_AGGS

    @classmethod
    def _remap_expr_name(cls, name: NarwhalsAggregation | Any, /) -> str:
        return cls._REMAP_AGGS.get(name, name)

    @classmethod
    def _leaf_name(cls, expr: DictExpr, /) -> NarwhalsAggregation | Any:
        return next(expr._metadata.op_nodes_reversed()).name

    @classmethod
    def _kwargs(cls, expr: DictExpr, /) -> dict[str, Any]:
        return next(expr._metadata.op_nodes_reversed()).kwargs

    def _group_indices(self, frame: DictDataFrame) -> dict[Any, list[int]]:
        """Map each key to its row indices, in order of first appearance."""
        return _bucket_by_keys([frame.native[key] for key in self._keys])

    @staticmethod
    def _leaf_order_by(leaf: Any) -> tuple[str, ...]:
        """The `order_by` an order-dependent leaf (`first`/`last`) sorts each group by."""
        if leaf.name in DictGroupBy._ORDER_DEPENDENT:
            return tuple(leaf.kwargs.get("order_by", ()))
        return ()

    @staticmethod
    def _agg_elementwise(
        expr: DictExpr,
        frame: DictDataFrame,
        *,
        groups: _GroupedRows,
        order_by: tuple[str, ...],
        output_names: Sequence[str],
        aliases: Sequence[str],
    ) -> dict[str, list[Any]]:
        """Aggregate an elementwise input, evaluated once over the whole frame.

        E.g. `col("a").sum()` or `(col("a") * col("b")).sum()`: the input gives one
        value per row, which is split by group and aggregated with a builtin if
        possible, or else with the `DictSeries` method.
        """
        assert expr._agg_source is not None  # noqa: S101
        source, method_name, kwargs = expr._agg_source
        # Groups are already sorted by `order_by`, which `first`/`last` would re-apply.
        kwargs = {name: value for name, value in kwargs.items() if name != "order_by"}
        version = frame._version
        result: dict[str, list[Any]] = {}
        # Multi-output inputs (e.g. `all()`) may include the group keys, which
        # `output_names` (named after the root columns) excludes.
        wanted = set(output_names)
        inputs = [
            series
            for series, root_name in zip(
                source(frame), source._evaluate_output_names(frame), strict=True
            )
            if root_name in wanted
        ]
        for series, alias in zip(inputs, aliases, strict=True):
            native = series.native
            values = (
                groups.gather(order_by, series.name)
                if frame.native.get(series.name) is native
                else groups.split(native, order_by)
            )
            fast = _NULL_SAFE_AGGS.get(method_name) or (
                _NULL_FREE_AGGS if None not in native else _NULL_SKIPPING_AGGS
            ).get(method_name)
            if fast is not None and not kwargs:
                result[alias] = list(map(fast, values))
            else:
                result[alias] = [
                    getattr(DictSeries(group, name=alias, version=version), method_name)(
                        **kwargs
                    )
                    for group in values
                ]
        return result

    @staticmethod
    def _agg_complex(
        expr: DictExpr, group_frames: Iterable[DictDataFrame], aliases: Sequence[str]
    ) -> dict[str, list[Any]]:
        """Fallback: evaluate the full expression against each group's sub-frame.

        Everything is in memory, so any expression whose last node aggregates is
        supported, e.g. `nw.col("b").round(2).mean()` or `(nw.col("b") * nw.col("c")).sum()`.
        Narwhals already rejects non-aggregating expressions at the API level.
        """
        result: dict[str, list[Any]] = {alias: [] for alias in aliases}
        for sub_frame in group_frames:
            evaluated = {series.name: series for series in expr(sub_frame)}
            for alias in aliases:
                series = evaluated[alias]
                if len(series) != 1:  # pragma: no cover
                    msg = (
                        f"Safety assertion failed: expected a single value per group for "
                        f"{alias!r}, got {len(series)} values."
                    )
                    raise AssertionError(msg)
                result[alias].append(series.native[0])
        return result

    def agg(self, *exprs: DictExpr) -> DictDataFrame:
        frame = self.compliant
        groups = _GroupedRows(frame, self._keys)
        result: DictFrame = {
            key_name: groups.key_column(position)
            for position, key_name in enumerate(self._keys)
        }
        exclude = (*self._keys, *self._output_key_names)
        for expr in exprs:
            output_names, aliases = evaluate_output_names_and_aliases(
                expr, frame, exclude
            )
            op_nodes = list(expr._metadata.op_nodes_reversed())
            order_by = self._leaf_order_by(op_nodes[0])
            if len(op_nodes) == 1 and op_nodes[0].name == "len":
                # `agg(nw.len())`: no input column, just count rows per group. Other
                # single-node exprs (e.g. `nw.corr(...)`) go through the fallback.
                result[aliases[0]] = [len(indices) for indices in groups.indices(())]
            elif expr._agg_source is not None and all(
                node.is_elementwise() for node in op_nodes[1:]
            ):
                # e.g. `col("a").sum()` or `(col("a") * 2).round(1).sum()`: one value
                # per row goes into the aggregation, so the input can be evaluated once.
                # Otherwise (e.g. `(col("a") - col("a").mean()).sum()`, `filter(...)`),
                # the fallback evaluates the whole expression on each group.
                result.update(
                    self._agg_elementwise(
                        expr,
                        frame,
                        groups=groups,
                        order_by=order_by,
                        output_names=output_names,
                        aliases=aliases,
                    )
                )
            else:
                columns = tuple(expr._pruned(frame).columns)
                result.update(
                    self._agg_complex(expr, groups.frames(order_by, columns), aliases)
                )

        return frame._with_native(result, validate_column_names=False).rename(
            dict(zip(self._keys, self._output_key_names, strict=True))
        )

    def __iter__(self) -> Iterator[tuple[tuple[Any, ...], DictDataFrame]]:
        frame = self.compliant
        single_key = len(self._keys) == 1
        selected = frame.simple_select(*self._df.columns)
        for key, indices in self._group_indices(frame).items():
            # Keys are always yielded as tuples, so wrap the single-key raw value.
            yield ((key,) if single_key else key), selected._gather_positions(indices)
