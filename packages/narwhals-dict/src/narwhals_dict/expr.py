from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

from narwhals._compliant import EagerExpr
from narwhals._expression_parsing import evaluate_output_names_and_aliases
from narwhals._utils import Implementation, generate_temporary_column_name
from narwhals_dict.series import DictSeries
from narwhals_dict.window import GROUPED_WINDOW_LEAVES, apply_window_kernel

if TYPE_CHECKING:
    from collections.abc import Sequence

    from typing_extensions import Self

    from narwhals._compliant.typing import AliasNames, EvalNames, EvalSeries
    from narwhals._utils import Version, _LimitedContext
    from narwhals_dict.dataframe import DictDataFrame
    from narwhals_dict.namespace import DictNamespace


class DictExpr(EagerExpr["DictDataFrame", DictSeries]):
    _implementation: Implementation = Implementation.UNKNOWN

    def __init__(
        self,
        call: EvalSeries[DictDataFrame, DictSeries],
        *,
        evaluate_output_names: EvalNames[DictDataFrame],
        alias_output_names: AliasNames | None,
        version: Version,
        implementation: Implementation = Implementation.UNKNOWN,
    ) -> None:
        self._call = call
        self._evaluate_output_names = evaluate_output_names
        self._alias_output_names = alias_output_names
        self._version = version

    @classmethod
    def from_column_names(
        cls,
        evaluate_column_names: EvalNames[DictDataFrame],
        /,
        *,
        context: _LimitedContext,
    ) -> Self:
        def func(df: DictDataFrame) -> list[DictSeries]:
            try:
                return [
                    DictSeries(df.native[name], name=name, version=df._version)
                    for name in evaluate_column_names(df)
                ]
            except KeyError as e:
                if error := df._check_columns_exist(evaluate_column_names(df)):
                    raise error from e
                raise

        return cls(
            func,
            evaluate_output_names=evaluate_column_names,
            alias_output_names=None,
            version=context._version,
        )

    @classmethod
    def from_column_indices(cls, *column_indices: int, context: _LimitedContext) -> Self:
        def func(df: DictDataFrame) -> list[DictSeries]:
            columns = df.columns
            return [
                DictSeries(df.native[columns[i]], name=columns[i], version=df._version)
                for i in column_indices
            ]

        return cls(
            func,
            evaluate_output_names=cls._eval_names_indices(column_indices),
            alias_output_names=None,
            version=context._version,
        )

    def __narwhals_namespace__(self) -> DictNamespace:
        from narwhals_dict.namespace import DictNamespace

        return DictNamespace(version=self._version)

    def ewm_mean(
        self,
        *,
        com: float | None,
        span: float | None,
        half_life: float | None,
        alpha: float | None,
        adjust: bool,
        min_samples: int,
        ignore_nulls: bool,
    ) -> Self:
        return self._reuse_series(
            "ewm_mean",
            com=com,
            span=span,
            half_life=half_life,
            alpha=alpha,
            adjust=adjust,
            min_samples=min_samples,
            ignore_nulls=ignore_nulls,
        )

    def _is_close_float_promote(self) -> Self:
        return self.cast(self._version.dtypes.Float64())

    def _over_without_partition_by(self, order_by: Sequence[str]) -> Self:
        # e.g. `nw.col('a').cum_sum().over(order_by=key)`: no grouping needed.
        assert order_by  # noqa: S101
        # NB: `is_scalar_like` here is the *whole-expression* property: an
        # all-aggregation/literal expression collapses to a single value that
        # `over` must broadcast back. `over` below instead branches on the *leaf
        # node* kind to pick a dispatch path. Both mirror the PyArrow backend.
        meta = self._metadata

        def func(df: DictDataFrame) -> Sequence[DictSeries]:
            version = df._version
            token = generate_temporary_column_name(8, df.columns)
            df = df.with_row_index(token, order_by=None).sort(
                *order_by, descending=False, nulls_last=False
            )
            results = self(df.drop([token], strict=True))
            if meta.is_scalar_like:
                # `over` is length-preserving, so broadcast the scalar result back.
                size = len(df)
                return [
                    DictSeries(
                        [s.native[0] if s.native else None] * size,
                        name=s.name,
                        version=version,
                    )
                    for s in results
                ]
            # Scatter the sorted results back into the original row order: `token`
            # holds each sorted row's original index, so `scatter` reverses the sort.
            token_series = df.get_column(token)
            return [s.scatter(token_series, s) for s in results]

        return self._from_callable(
            func,
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            context=self,
        )

    @staticmethod
    def _partition_indices(
        df: DictDataFrame, partition_by: Sequence[str], order_by: Sequence[str]
    ) -> dict[Any, list[int]]:
        """Group row indices by partition key, each group in `order_by` order.

        One global sort (ascending, nulls first) replaces the per-group sorts:
        iterating rows in sorted order and bucketing by key yields groups whose
        indices are already ordered. Null keys work natively. Single-key
        partitions skip the per-row tuple, mirroring `_join_keys`.
        """
        key_columns = [df.native[name] for name in partition_by]
        order: Sequence[int] = (
            df._sorted_indices(order_by, descending=False, nulls_last=False)
            if order_by
            else range(len(df))
        )
        groups: dict[Any, list[int]] = {}

        keys = (
            key_columns[0]
            if len(key_columns) == 1
            else list(zip(*key_columns, strict=True))
        )
        for index in order:
            groups.setdefault(keys[index], []).append(index)
        return groups

    def _evaluate_per_partition(
        self,
        df: DictDataFrame,
        partition_by: Sequence[str],
        order_by: Sequence[str],
        *,
        broadcast: bool,
    ) -> Sequence[DictSeries]:
        """Evaluate the whole expression once per partition, in `order_by` order.

        Each result is scattered back to the partition's original row positions.
        """
        groups = self._partition_indices(df, partition_by, order_by)
        if not groups:
            # Empty frame: evaluate once so output names and dtypes are right.
            return self(df)

        # Column pruning: a `col`-rooted chain with no expressifiable arguments
        # reads only its root columns, so gather just those into each sub-frame
        # instead of every column. Any node carrying `exprs` (e.g.
        # `fill_null(nw.col("b").mean())`) may reference other columns, so keep
        # the whole frame in that case.
        op_nodes = list(self._metadata.op_nodes_reversed())
        if op_nodes[-1].name == "col" and not any(node.exprs for node in op_nodes):
            source = df.simple_select(*self._evaluate_output_names(df))
        else:
            source = df

        num_rows = len(df)
        outputs: list[list[Any]] = []
        names: list[str] = []
        for indices in groups.values():
            results = self(source._gather_positions(indices))
            if not outputs:
                outputs = [[None] * num_rows for _ in results]
                names = [s.name for s in results]
            for out, series in zip(outputs, results, strict=True):
                values = series.native
                if broadcast:
                    value = values[0] if values else None
                    for i in indices:
                        out[i] = value
                else:
                    for i, value in zip(indices, values, strict=True):
                        out[i] = value
        version = df._version
        return [
            DictSeries(values, name=name, version=version)
            for name, values in zip(names, outputs, strict=True)
        ]

    def _over_grouped_kernel(
        self, partition_by: Sequence[str], order_by: Sequence[str]
    ) -> Self | None:
        """Fast path for `col(name).<window_leaf>(**kwargs).over(partition_by)`.

        Returns `None` (fall back to `_evaluate_per_partition`) unless the whole
        expression is a single supported window leaf reading stored columns as-is.
        Mirrors `DictGroupBy.agg`'s: exactly leaf + `col` root, nothing in between
        (a transforming node like `when/then` or arithmetic must take the general path),
        and no expression-valued leaf kwargs.
        """
        op_nodes = list(self._metadata.op_nodes_reversed())
        if not (
            len(op_nodes) == 2
            and op_nodes[0].name in GROUPED_WINDOW_LEAVES
            and op_nodes[1].name == "col"
        ):
            return None
        leaf = op_nodes[0]
        if any(self._is_expr(value) for value in leaf.kwargs.values()):
            return None
        method_name, kwargs = leaf.name, leaf.kwargs

        def func(df: DictDataFrame) -> Sequence[DictSeries]:
            if not (groups := self._partition_indices(df, partition_by, order_by)):
                return self(df)
            output_names, aliases = evaluate_output_names_and_aliases(self, df, [])
            version = df._version
            kernel = partial(
                apply_window_kernel,
                method_name=method_name,
                groups=groups,
                num_rows=len(df),
                kwargs=kwargs,
                version=version,
            )
            return [
                DictSeries(kernel(column=df.native[name]), name=alias, version=version)
                for name, alias in zip(output_names, aliases, strict=True)
            ]

        return self._from_callable(
            func,
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            context=self,
        )

    def over(self, partition_by: Sequence[str], order_by: Sequence[str]) -> Self:
        if not partition_by:
            return self._over_without_partition_by(order_by)
        # Branch on the *leaf node* kind (not the whole-expression `is_scalar_like`
        # used in `_over_without_partition_by`): a scalar-like leaf (`sum`, `mean`,
        # ...) is one value per partition and takes the fast group-by path below;
        # anything else is a window leaf (one value per row) evaluated per partition.
        if not self._metadata.current_node.kind.is_scalar_like:
            # Window leaf (e.g. `cum_sum`, `fill_null(strategy=...)`, `rank`):
            # one value per row, so a group-by cannot represent the result.
            # Try the streaming grouped kernel first; it falls back to the
            # general per-partition evaluation for anything it cannot handle.
            if (fast := self._over_grouped_kernel(partition_by, order_by)) is not None:
                return fast

            def window_func(df: DictDataFrame) -> Sequence[DictSeries]:
                return self._evaluate_per_partition(
                    df, partition_by, order_by, broadcast=False
                )

            return self._from_callable(
                window_func,
                evaluate_output_names=self._evaluate_output_names,
                alias_output_names=self._alias_output_names,
                context=self,
            )

        # Aggregating leaf (e.g. `nw.col('a').sum().over('b')`): one value per
        # partition, computed with a single `group_by` pass and broadcast back to
        # each original row via a plain dict lookup (the eager equivalent of
        # arrow's group_by + left join, minus the join).
        def func(df: DictDataFrame) -> Sequence[DictSeries]:
            _, aliases = evaluate_output_names_and_aliases(self, df, [])

            if set(aliases).intersection(partition_by):
                # E.g. `df.with_columns(nw.all().sum().over('a'))`: the group-by
                # output would collide with the key columns, but per-partition
                # evaluation has no such clash. The leaf is scalar-like here, so
                # broadcast its single value per partition.
                return self._evaluate_per_partition(
                    df, partition_by, order_by, broadcast=True
                )

            version = df._version
            # Broadcast target: each input row's partition key, in original order.
            # `None` keys are handled natively since Python tuples are hashable.
            original_keys = list(
                zip(*(df.native[name] for name in partition_by), strict=True)
            )
            grouping = (
                df.sort(*order_by, descending=False, nulls_last=False) if order_by else df
            )
            tmp = grouping.group_by(partition_by, drop_null_keys=False).agg(self)
            tmp_keys = list(
                zip(*(tmp.native[name] for name in partition_by), strict=True)
            )
            lookups = {
                alias: dict(zip(tmp_keys, tmp.native[alias], strict=True))
                for alias in aliases
            }
            return [
                DictSeries(
                    [lookups[alias][key] for key in original_keys],
                    name=alias,
                    version=version,
                )
                for alias in aliases
            ]

        return self._from_callable(
            func,
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            context=self,
        )
