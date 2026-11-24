from __future__ import annotations

from itertools import chain
from typing import TYPE_CHECKING, Any, ClassVar, cast

import dask.dataframe as dd

from narwhals.compliant import CompliantGroupBy
from narwhals_dask._vendored import (
    evaluate_aliases,
    evaluate_output_names_and_aliases,
    is_sequence_of,
    make_group_by_kwargs,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from typing import TypeAlias

    import dask.dataframe.dask_expr as dx
    import pandas as pd
    from pandas.core.groupby import SeriesGroupBy as _PandasSeriesGroupBy

    from narwhals_dask._vendored import NarwhalsAggregation
    from narwhals_dask.dataframe import DaskLazyFrame
    from narwhals_dask.expr import DaskExpr

    PandasSeriesGroupBy: TypeAlias = _PandasSeriesGroupBy[Any, Any]
    _AggFn: TypeAlias = Callable[..., Any]

Aggregation: TypeAlias = "str | _AggFn"
"""The name of an aggregation function, or the function itself."""


def n_unique() -> dd.Aggregation:
    def chunk(s: PandasSeriesGroupBy) -> pd.Series[Any]:
        return s.nunique(dropna=False)

    def agg(s0: PandasSeriesGroupBy) -> pd.Series[Any]:
        return s0.sum()

    return dd.Aggregation(name="nunique", chunk=chunk, agg=agg)


def _all() -> dd.Aggregation:
    def chunk(s: PandasSeriesGroupBy) -> pd.Series[Any]:
        return s.all(skipna=True)

    def agg(s0: PandasSeriesGroupBy) -> pd.Series[Any]:
        return s0.all(skipna=True)

    return dd.Aggregation(name="all", chunk=chunk, agg=agg)


def _any() -> dd.Aggregation:
    def chunk(s: PandasSeriesGroupBy) -> pd.Series[Any]:
        return s.any(skipna=True)

    def agg(s0: PandasSeriesGroupBy) -> pd.Series[Any]:
        return s0.any(skipna=True)

    return dd.Aggregation(name="any", chunk=chunk, agg=agg)


def _var_or_std(ddof: int, *, sqrt: bool) -> dd.Aggregation:
    # Dask's `GroupBy.var` gives `inf` instead of null when N < ddof, and `GroupBy.std`
    # can round a constant group's variance below zero, giving NaN.
    def chunk(s: PandasSeriesGroupBy) -> tuple[pd.Series[Any], ...]:
        n, mean = s.count(), s.mean()
        return n, mean * n, (s.var(ddof=0) + mean * mean) * n

    def agg(*chunks: PandasSeriesGroupBy) -> tuple[pd.Series[Any], ...]:
        return tuple(c.sum() for c in chunks)

    def finalize(
        n: pd.Series[Any], total: pd.Series[Any], sum_squares: pd.Series[Any]
    ) -> pd.Series[Any]:
        m2 = (sum_squares - total * total / n).clip(lower=0)
        result = (m2 / (n - ddof)).where(n > ddof)
        return result**0.5 if sqrt else result

    name = f"{'std' if sqrt else 'var'}_ddof{ddof}"
    return dd.Aggregation(name=name, chunk=chunk, agg=agg, finalize=finalize)


def var(ddof: int) -> dd.Aggregation:
    return _var_or_std(ddof, sqrt=False)


def std(ddof: int) -> dd.Aggregation:
    return _var_or_std(ddof, sqrt=True)


class DaskLazyGroupBy(CompliantGroupBy["DaskLazyFrame", "DaskExpr"]):
    _REMAP_AGGS: ClassVar[Mapping[NarwhalsAggregation, Aggregation]] = {
        "sum": "sum",
        "mean": "mean",
        "median": "median",
        "max": "max",
        "min": "min",
        "std": std,
        "var": var,
        "len": "size",
        "n_unique": n_unique,
        "count": "count",
        "quantile": "quantile",
        "all": _all,
        "any": _any,
    }

    def __init__(
        self,
        df: DaskLazyFrame,
        keys: Sequence[DaskExpr] | Sequence[str],
        /,
        *,
        drop_null_keys: bool,
    ) -> None:
        self._compliant_frame, self._keys, self._output_key_names = self._parse_keys(
            df, keys=keys
        )
        group_by_kwargs = make_group_by_kwargs(drop_null_keys=drop_null_keys)
        self._grouped = self.compliant.native.groupby(self._keys, **group_by_kwargs)

    def agg(self, *exprs: DaskExpr) -> DaskLazyFrame:
        from narwhals_dask.dataframe import DaskLazyFrame

        if not exprs:
            # No aggregation provided
            return (
                self.compliant.simple_select(*self._keys)
                .unique(self._keys, keep="any", order_by=None)
                .rename(dict(zip(self._keys, self._output_key_names, strict=False)))
            )

        self._ensure_all_simple(exprs)
        # This should be the fastpath, but cuDF is too far behind to use it.
        # - https://github.com/rapidsai/cudf/issues/15118
        # - https://github.com/rapidsai/cudf/issues/15084
        simple_aggregations: dict[str, tuple[str, Aggregation]] = {}
        exclude = (*self._keys, *self._output_key_names)
        for expr in exprs:
            output_names, aliases = evaluate_output_names_and_aliases(
                expr, self.compliant, exclude
            )
            nodes = list(expr._metadata.op_nodes_reversed())
            last_node = nodes[0]
            if len(nodes) == 1:
                # e.g. `agg(nw.len())`
                column = self._keys[0]
                agg_fn = self._remap_expr_name(last_node.name)
                simple_aggregations.update(dict.fromkeys(aliases, (column, agg_fn)))
                continue

            # e.g. `agg(nw.mean('a'))`
            agg_fn = self._remap_expr_name(self._leaf_name(expr))
            # deal with n_unique case in a "lazy" mode to not depend on dask globally
            agg_fn = agg_fn(**last_node.kwargs) if callable(agg_fn) else agg_fn
            simple_aggregations.update(
                (alias, (output_name, agg_fn))
                for alias, output_name in zip(aliases, output_names, strict=True)
            )
        agged = cast("dx.DataFrame", self._grouped.agg(**simple_aggregations))
        return DaskLazyFrame(agged.reset_index(), version=self.compliant._version).rename(
            dict(zip(self._keys, self._output_key_names, strict=False))
        )

    def _parse_keys(
        self, compliant_frame: DaskLazyFrame, keys: Sequence[DaskExpr] | Sequence[str]
    ) -> tuple[DaskLazyFrame, list[str], list[str]]:
        if is_sequence_of(keys, str):
            keys_str = list(keys)
            return compliant_frame, keys_str, keys_str.copy()
        return self._parse_expr_keys(compliant_frame, keys=keys)

    @staticmethod
    def _parse_expr_keys(
        compliant_frame: DaskLazyFrame, keys: Sequence[DaskExpr]
    ) -> tuple[DaskLazyFrame, list[str], list[str]]:
        """Evaluate key expressions into temporary columns which don't clash with existing ones.

        The output key names are returned too, so the temporary columns can be renamed
        back after aggregating, and excluded from unnamed expressions in `.agg(...)`.
        """
        tmp_name_length = max(len(str(c)) for c in compliant_frame.columns) + 1

        def _temporary_name(key: str) -> str:
            # 5 is the length of `__tmp`
            key_str = str(key)  # pandas allows non-string column names :sob:
            return f"_{key_str}_tmp{'_' * (tmp_name_length - len(key_str) - 5)}"

        keys_aliases = [expr._evaluate_aliases(compliant_frame) for expr in keys]
        safe_keys = [
            # multi-output expression cannot have duplicate names, hence it's safe to suffix
            key.name.map(_temporary_name)
            if (metadata := key._metadata) and metadata.expansion_kind.is_multi_output()
            # otherwise it's single named and we can use Expr.alias
            else key.alias(_temporary_name(new_names[0]))
            for key, new_names in zip(keys, keys_aliases, strict=True)
        ]
        return (
            compliant_frame.with_columns(*safe_keys),
            evaluate_aliases(compliant_frame, safe_keys),
            list(chain.from_iterable(keys_aliases)),
        )

    def _ensure_all_simple(self, exprs: Sequence[DaskExpr]) -> None:
        for expr in exprs:
            if not self._is_simple(expr):
                msg = (
                    "Non-trivial complex aggregation found.\n\n"
                    "Hint: you were probably trying to apply a non-elementary aggregation with a"
                    "'dask' table.\n"
                    "Please rewrite your query such that group-by aggregations "
                    "are elementary. For example, instead of:\n\n"
                    "    df.group_by('a').agg(nw.col('b').round(2).mean())\n\n"
                    "use:\n\n"
                    "    df.with_columns(nw.col('b').round(2)).group_by('a').agg(nw.col('b').mean())\n\n"
                )
                raise ValueError(msg)

    @classmethod
    def _is_simple(cls, expr: DaskExpr, /) -> bool:
        return expr._is_elementary() and cls._leaf_name(expr) in cls._REMAP_AGGS

    @classmethod
    def _remap_expr_name(cls, name: NarwhalsAggregation | Any, /) -> Aggregation:
        return cls._REMAP_AGGS.get(name, name)

    @classmethod
    def _leaf_name(cls, expr: DaskExpr, /) -> NarwhalsAggregation | Any:
        return next(expr._metadata.op_nodes_reversed()).name
