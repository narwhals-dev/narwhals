from __future__ import annotations

from operator import attrgetter
from typing import TYPE_CHECKING, Any, ClassVar

from narwhals._compliant import EagerGroupBy
from narwhals._expression_parsing import evaluate_output_names_and_aliases
from narwhals_dict.series import DictSeries

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

    from narwhals._compliant.typing import NarwhalsAggregation
    from narwhals_dict.dataframe import DictDataFrame
    from narwhals_dict.expr import DictExpr
    from narwhals_dict.typing import DictFrame


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

    def _group_indices(self, frame: DictDataFrame) -> dict[tuple[Any, ...], list[int]]:
        """Map each key tuple to its row indices, in order of first appearance."""
        key_columns = [frame.native[key] for key in self._keys]
        groups: dict[tuple[Any, ...], list[int]] = {}
        for index, key in enumerate(zip(*key_columns, strict=True)):
            groups.setdefault(key, []).append(index)
        return groups

    def _order_by(self, *exprs: DictExpr) -> tuple[str, ...]:
        """Collect a single `order_by` from order-dependent aggregations, like `first`/`last`."""
        order_by: tuple[str, ...] = ()
        for expr in exprs:
            node = next(expr._metadata.op_nodes_reversed())
            if node.name not in self._ORDER_DEPENDENT:
                continue
            if current := tuple(node.kwargs.get("order_by", ())):
                if order_by and current != order_by:
                    msg = f"Only one `order_by` can be specified in `group_by`. Found both {order_by} and {current}."
                    raise NotImplementedError(msg)
                order_by = current
        return order_by

    def _agg_simple(
        self,
        expr: DictExpr,
        frame: DictDataFrame,
        gather: Callable[[str], list[list[Any]]],
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
                for values in gather(output_name)
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
        if order_by := self._order_by(*exprs):
            frame = frame.sort(*order_by, descending=False, nulls_last=False)
        groups = self._group_indices(frame)
        group_indices = list(groups.values())

        # Each column is gathered by group indices at most once per `agg` call,
        # no matter how many expressions reference it. Sharing the gathered lists
        # (rather than copying) is safe: series and frame operations never mutate
        # native lists in place.
        gathered: dict[str, list[list[Any]]] = {}

        def gather(name: str) -> list[list[Any]]:
            if (columns := gathered.get(name)) is None:
                column = frame.native[name]
                columns = gathered[name] = [
                    [column[i] for i in indices] for indices in group_indices
                ]
            return columns

        result: DictFrame = {
            key_name: [key[position] for key in groups]
            for position, key_name in enumerate(self._keys)
        }

        exclude = (*self._keys, *self._output_key_names)
        # Sub-frames are only materialized if some expression needs the fallback,
        # and are shared across all such expressions.
        group_frames: list[DictDataFrame] | None = None
        for expr in exprs:
            output_names, aliases = evaluate_output_names_and_aliases(
                expr, frame, exclude
            )
            if len(list(expr._metadata.op_nodes_reversed())) == 1:
                # e.g. `agg(nw.len())`: no input column, just count rows per group.
                result[aliases[0]] = [len(indices) for indices in group_indices]
            elif self._is_simple(expr):
                result.update(
                    self._agg_simple(expr, frame, gather, output_names, aliases)
                )
            else:
                if group_frames is None:
                    group_frames = [
                        frame._with_native(
                            {name: gather(name)[position] for name in frame.native},
                            validate_column_names=False,
                        )
                        for position in range(len(group_indices))
                    ]
                result.update(self._agg_complex(expr, group_frames, aliases))

        return frame._with_native(result, validate_column_names=False).rename(
            dict(zip(self._keys, self._output_key_names, strict=True))
        )

    def __iter__(self) -> Iterator[tuple[Any, DictDataFrame]]:
        frame = self.compliant
        for key, indices in self._group_indices(frame).items():
            yield key, frame._gather(indices).simple_select(*self._df.columns)
