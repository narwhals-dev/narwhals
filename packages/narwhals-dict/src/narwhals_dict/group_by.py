from __future__ import annotations

from operator import attrgetter
from typing import TYPE_CHECKING, Any, ClassVar

from narwhals._compliant import EagerGroupBy
from narwhals._expression_parsing import evaluate_output_names_and_aliases
from narwhals_dict.series import DictSeries

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

    from narwhals._compliant.typing import NarwhalsAggregation
    from narwhals._expression_parsing import ExprNode
    from narwhals_dict.dataframe import DictDataFrame
    from narwhals_dict.expr import DictExpr
    from narwhals_dict.typing import DictFrame


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
    for index, key in it:
        groups.setdefault(key, []).append(index)
    return groups


class _GroupedRows:
    """Per-`agg` cache of grouped row indices, gathered columns, and sub-frames.

    `first`/`last` may each request a *different* `order_by`.

    Each column is gathered by group indices at most once per `agg` call,
    no matter how many expressions reference it. Sharing the gathered lists
    (rather than copying) is safe: series and frame operations never mutate
    native lists in place.
    """

    def __init__(self, frame: DictDataFrame, keys: Sequence[str]) -> None:
        self._frame = frame
        self._key_columns = [frame.native[key] for key in keys]
        # Keys are raw values for a single key, value tuples for several -- see
        # `_bucket_by_keys`. `row_keys` uses the matching representation so the
        # `order_by` re-bucketing below looks them up consistently.
        base_groups = _bucket_by_keys(self._key_columns)
        self.keys: list[Any] = list(base_groups)
        self._indices: dict[tuple[str, ...], list[list[int]]] = {
            (): [base_groups[key] for key in self.keys]
        }

        # Row keys are only needed to re-bucket a non-trivial `order_by`, so they are built lazily
        self._row_keys: Sequence[Any] | None = None
        self._columns: dict[tuple[tuple[str, ...], str], list[list[Any]]] = {}
        self._frames: dict[tuple[str, ...], list[DictDataFrame]] = {}

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
        return (
            list(self.keys)
            if len(self._key_columns) == 1
            else [key[position] for key in self.keys]
        )

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
            column = self._frame.native[name]
            columns = self._columns[cache_key] = [
                [column[i] for i in indices] for indices in self.indices(order_by)
            ]
        return columns

    def frames(self, order_by: tuple[str, ...]) -> list[DictDataFrame]:
        """One sub-frame per group (all columns), rows in `order_by` order."""
        if (frames := self._frames.get(order_by)) is None:
            col_names = self._frame.columns
            frames = self._frames[order_by] = [
                self._frame._with_native(
                    {name: self.gather(order_by, name)[position] for name in col_names},
                    validate_column_names=False,
                )
                for position in range(len(self.keys))
            ]
        return frames


class DictGroupBy(EagerGroupBy["DictDataFrame", "DictExpr", str]):
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

    def _group_indices(self, frame: DictDataFrame) -> dict[Any, list[int]]:
        """Map each key to its row indices, in order of first appearance."""
        return _bucket_by_keys([frame.native[key] for key in self._keys])

    @staticmethod
    def _leaf_order_by(leaf: ExprNode) -> tuple[str, ...]:
        """The `order_by` an order-dependent leaf (`first`/`last`) sorts each group by."""
        if leaf.name in DictGroupBy._ORDER_DEPENDENT:
            return tuple(leaf.kwargs.get("order_by", ()))
        return ()

    def _agg_simple(
        self,
        expr: DictExpr,
        frame: DictDataFrame,
        order_by: tuple[str, ...],
        gather: Callable[[tuple[str, ...], str], list[list[Any]]],
        output_names: Sequence[str],
        aliases: Sequence[str],
    ) -> dict[str, list[Any]]:
        """Fast path: dispatch elementary aggregations straight to `DictSeries` methods."""
        method = attrgetter(self._remap_expr_name(self._leaf_name(expr)))
        version = frame._version
        kwargs = {
            name: value
            for name, value in self._kwargs(expr).items()
            if name != "order_by"
        }

        return {
            alias: [
                method(DictSeries(values, name=output_name, version=version))(**kwargs)
                for values in gather(order_by, output_name)
            ]
            for output_name, alias in zip(output_names, aliases, strict=True)
        }

    @staticmethod
    def _agg_complex(
        expr: DictExpr, group_frames: Sequence[DictDataFrame], aliases: Sequence[str]
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
            if len(op_nodes) == 1:
                # e.g. `agg(nw.len())`: no input column, just count rows per group.
                result[aliases[0]] = [len(indices) for indices in groups.indices(())]
            elif self._is_simple(expr) and op_nodes[1].name == "col":
                # e.g. `col("a").sum()`: the aggregation applies directly to a stored
                # column, so we can gather it and dispatch to a `DictSeries` method.
                # A non-`col` node before the leaf (e.g. `when(...).then(...)`,
                # `round(...)`, arithmetic) transforms the values first and must go
                # through the fallback, which re-evaluates the whole expression.
                result.update(
                    self._agg_simple(
                        expr, frame, order_by, groups.gather, output_names, aliases
                    )
                )
            else:
                result.update(self._agg_complex(expr, groups.frames(order_by), aliases))

        return frame._with_native(result, validate_column_names=False).rename(
            dict(zip(self._keys, self._output_key_names, strict=True))
        )

    def __iter__(self) -> Iterator[tuple[tuple[Any, ...], DictDataFrame]]:
        frame = self.compliant
        single_key = len(self._keys) == 1
        for key, indices in self._group_indices(frame).items():
            # Keys are always yielded as tuples, so wrap the single-key raw value.
            yield (
                (key,) if single_key else key,
                frame._gather(indices).simple_select(*self._df.columns),
            )
