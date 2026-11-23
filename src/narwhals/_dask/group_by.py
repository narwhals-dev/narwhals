from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, cast

import dask.dataframe as dd

from narwhals._compliant import DepthTrackingGroupBy
from narwhals._dask.utils import make_group_by_kwargs
from narwhals._expression_parsing import evaluate_output_names_and_aliases

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from typing import TypeAlias

    import dask_expr as dx
    import pandas as pd
    from pandas.core.groupby import SeriesGroupBy as _PandasSeriesGroupBy

    from narwhals._compliant.typing import NarwhalsAggregation
    from narwhals._dask.dataframe import DaskLazyFrame
    from narwhals._dask.expr import DaskExpr

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


class DaskLazyGroupBy(DepthTrackingGroupBy["DaskLazyFrame", "DaskExpr", Aggregation]):
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
        from narwhals._dask.dataframe import DaskLazyFrame

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
