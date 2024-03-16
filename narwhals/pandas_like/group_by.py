from __future__ import annotations

import collections
import os
from copy import copy
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Iterable

from narwhals.pandas_like.utils import dataframe_from_dict
from narwhals.pandas_like.utils import evaluate_simple_aggregation
from narwhals.pandas_like.utils import horizontal_concat
from narwhals.pandas_like.utils import is_simple_aggregation
from narwhals.pandas_like.utils import item
from narwhals.pandas_like.utils import parse_into_exprs
from narwhals.utils import remove_prefix

if TYPE_CHECKING:
    from narwhals.pandas_like.dataframe import PandasDataFrame
    from narwhals.pandas_like.expr import PandasExpr
    from narwhals.pandas_like.typing import IntoPandasExpr


class PandasGroupBy:
    def __init__(self, df: PandasDataFrame, keys: list[str]) -> None:
        self._df = df
        self._keys = list(keys)

    def agg(
        self,
        *aggs: IntoPandasExpr | Iterable[IntoPandasExpr],
        **named_aggs: IntoPandasExpr,
    ) -> PandasDataFrame:
        df = self._df._dataframe
        exprs = parse_into_exprs(
            self._df._implementation,
            *aggs,
            **named_aggs,
        )
        grouped = df.groupby(
            list(self._keys),
            sort=False,
            as_index=False,
        )
        implementation: str = self._df._implementation
        output_names: list[str] = copy(self._keys)
        for expr in exprs:
            if expr._output_names is None:
                msg = (
                    "Anonymous expressions are not supported in group_by.agg.\n"
                    "Instead of `pl.all()`, try using a named expression, such as "
                    "`pl.col('a', 'b')`\n"
                )
                raise ValueError(msg)
            output_names.extend(expr._output_names)

        if implementation == "pandas" and not os.environ.get("NARWHALS_FORCE_GENERIC"):
            return agg_pandas(
                grouped,
                exprs,
                self._keys,
                output_names,
                self._from_dataframe,
            )
        return agg_generic(
            grouped,
            exprs,
            self._keys,
            output_names,
            implementation,
            self._from_dataframe,
        )

    def _from_dataframe(self, df: PandasDataFrame) -> PandasDataFrame:
        from narwhals.pandas_like.dataframe import PandasDataFrame

        return PandasDataFrame(
            df,
            implementation=self._df._implementation,
        )


def agg_pandas(
    grouped: Any,
    exprs: list[PandasExpr],
    keys: list[str],
    output_names: list[str],
    from_dataframe: Callable[[Any], PandasDataFrame],
) -> PandasDataFrame:
    """
    This should be the fastpath, but cuDF is too far behind to use it.

    - https://github.com/rapidsai/cudf/issues/15118
    - https://github.com/rapidsai/cudf/issues/15084
    """
    import pandas as pd

    simple_aggs = []
    complex_aggs = []
    for expr in exprs:
        if is_simple_aggregation(expr):
            simple_aggs.append(expr)
        else:
            complex_aggs.append(expr)
    simple_aggregations = {}
    for expr in simple_aggs:
        assert expr._root_names is not None
        assert expr._output_names is not None
        for root_name, output_name in zip(expr._root_names, expr._output_names):
            name = remove_prefix(expr._function_name, "col->")
            simple_aggregations[output_name] = pd.NamedAgg(column=root_name, aggfunc=name)
    result_simple = grouped.agg(**simple_aggregations) if simple_aggregations else None

    def func(df: Any) -> Any:
        out_group = []
        out_names = []
        for expr in complex_aggs:
            results_keys = expr._call(from_dataframe(df))
            for result_keys in results_keys:
                out_group.append(item(result_keys._series))
                out_names.append(result_keys.name)
        return pd.Series(out_group, index=out_names)

    result_complex = grouped.apply(func, include_groups=False)

    if result_simple is not None:
        result = pd.concat(
            [result_simple, result_complex.drop(columns=keys)], axis=1, copy=False
        )
    else:
        result = result_complex
    return from_dataframe(result.loc[:, output_names])


def agg_generic(  # noqa: PLR0913
    grouped: Any,
    exprs: list[PandasExpr],
    group_by_keys: list[str],
    output_names: list[str],
    implementation: str,
    from_dataframe: Callable[[Any], PandasDataFrame],
) -> PandasDataFrame:
    dfs: list[Any] = []
    to_remove: list[int] = []
    for i, expr in enumerate(exprs):
        if is_simple_aggregation(expr):
            dfs.append(evaluate_simple_aggregation(expr, grouped))
            to_remove.append(i)
    exprs = [expr for i, expr in enumerate(exprs) if i not in to_remove]

    out: dict[str, list[Any]] = collections.defaultdict(list)
    for keys, df_keys in grouped:
        for key, name in zip(keys, group_by_keys):
            out[name].append(key)
        for expr in exprs:
            results_keys = expr._call(from_dataframe(df_keys))
            for result_keys in results_keys:
                out[result_keys.name].append(result_keys.item())

    results_keys = dataframe_from_dict(out, implementation=implementation)
    results_df = horizontal_concat(
        [results_keys, *dfs], implementation=implementation
    ).loc[:, output_names]
    return from_dataframe(results_df)
