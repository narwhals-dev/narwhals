from __future__ import annotations

import collections
import warnings
from copy import copy
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Iterable

from narwhals.pandas_like.utils import is_simple_aggregation
from narwhals.pandas_like.utils import item
from narwhals.pandas_like.utils import parse_into_exprs
from narwhals.utils import parse_version
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
            as_index=True,
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

        return agg_pandas(
            grouped,
            exprs,
            self._keys,
            output_names,
            self._from_dataframe,
            implementation,
        )

    def _from_dataframe(self, df: PandasDataFrame) -> PandasDataFrame:
        from narwhals.pandas_like.dataframe import PandasDataFrame

        return PandasDataFrame(
            df,
            implementation=self._df._implementation,
        )


def agg_pandas(  # noqa: PLR0913,PLR0915
    grouped: Any,
    exprs: list[PandasExpr],
    keys: list[str],
    output_names: list[str],
    from_dataframe: Callable[[Any], PandasDataFrame],
    implementation: Any,
) -> PandasDataFrame:
    """
    This should be the fastpath, but cuDF is too far behind to use it.

    - https://github.com/rapidsai/cudf/issues/15118
    - https://github.com/rapidsai/cudf/issues/15084
    """
    import pandas as pd

    from narwhals.pandas_like.namespace import PandasNamespace

    simple_aggs = []
    complex_aggs = []
    for expr in exprs:
        if is_simple_aggregation(expr):
            simple_aggs.append(expr)
        else:
            complex_aggs.append(expr)
    simple_aggregations = {}
    for expr in simple_aggs:
        if expr._depth == 0:
            # e.g. agg(pl.len())
            assert expr._output_names is not None
            for output_name in expr._output_names:
                simple_aggregations[output_name] = (
                    keys[0],
                    expr._function_name.replace("len", "size"),
                )
            continue

        assert expr._root_names is not None
        assert expr._output_names is not None
        for root_name, output_name in zip(expr._root_names, expr._output_names):
            name = remove_prefix(expr._function_name, "col->")
            simple_aggregations[output_name] = (root_name, name)

    if simple_aggregations:
        aggs = collections.defaultdict(list)
        name_mapping = {}
        for output_name, named_agg in simple_aggregations.items():
            aggs[named_agg[0]].append(named_agg[1])
            name_mapping[f"{named_agg[0]}_{named_agg[1]}"] = output_name
        try:
            result_simple = grouped.agg(aggs)
        except AttributeError as exc:
            raise RuntimeError(
                "Failed to aggregated - does your aggregation function return a scalar?"
            ) from exc
        result_simple.columns = [f"{a}_{b}" for a, b in result_simple.columns]
        result_simple = result_simple.rename(columns=name_mapping).reset_index()
    else:
        result_simple = None

    plx = PandasNamespace(implementation=implementation)

    def func(df: Any) -> Any:
        out_group = []
        out_names = []
        for expr in complex_aggs:
            results_keys = expr._call(from_dataframe(df))
            for result_keys in results_keys:
                out_group.append(item(result_keys._series))
                out_names.append(result_keys.name)
        return plx.make_native_series(name="", data=out_group, index=out_names)

    if complex_aggs:
        warnings.warn(
            "Found complex group-by expression, which can't be expressed efficiently with the "
            "pandas API. If you can, please rewrite your query such that group-by aggregations "
            "are simple (e.g. mean, std, min, max, ...).",
            UserWarning,
            stacklevel=2,
        )
        if implementation == "pandas":
            import pandas as pd

            if parse_version(pd.__version__) < parse_version("2.2.0"):  # pragma: no cover
                result_complex = grouped.apply(func)
            else:
                result_complex = grouped.apply(func, include_groups=False)
        else:  # pragma: no cover
            result_complex = grouped.apply(func)

    if result_simple is not None and not complex_aggs:
        result = result_simple
    elif result_simple is not None and complex_aggs:
        result = pd.concat(
            [result_simple, result_complex.reset_index(drop=True)],
            axis=1,
            copy=False,
        )
    elif complex_aggs:
        result = result_complex.reset_index()
    else:
        raise AssertionError("At least one aggregation should have been passed")
    return from_dataframe(result.loc[:, output_names])
