from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from narwhals.dependencies import get_dask_expr

if TYPE_CHECKING:
    from narwhals._dask.dataframe import DaskLazyFrame


def maybe_evaluate(df: DaskLazyFrame, obj: Any) -> Any:
    from narwhals._dask.expr import DaskExpr

    if isinstance(obj, DaskExpr):
        results = obj._call(df)
        if len(results) != 1:  # pragma: no cover
            msg = "Multi-output expressions not supported in this context"
            raise NotImplementedError(msg)
        result = results[0]
        if not get_dask_expr()._expr.are_co_aligned(
            df._native_dataframe._expr, result._expr
        ):  # pragma: no cover
            # are_co_aligned is a method which cheaply checks if two Dask expressions
            # have the same index, and therefore don't require index alignment.
            # If someone only operates on a Dask DataFrame via expressions, then this
            # should always be the case: expression outputs (by definition) all come from the
            # same input dataframe, and Dask Series does not have any operations which
            # change the index. Nonetheless, we perform this safety check anyway.

            # However, we still need to carefully vet which methods we support for Dask, to
            # avoid issues where `are_co_aligned` doesn't do what we want it to do:
            # https://github.com/dask/dask-expr/issues/1112.
            msg = "Implicit index alignment is not support for Dask DataFrame in Narwhals"
            raise NotImplementedError(msg)
        return result
    return obj


def parse_exprs_and_named_exprs(
    df: DaskLazyFrame, *exprs: Any, **named_exprs: Any
) -> dict[str, Any]:
    results = {}
    for expr in exprs:
        if hasattr(expr, "__narwhals_expr__"):
            _results = expr._call(df)
            _names = expr._output_names
        elif isinstance(expr, str):
            _results = [df._native_dataframe.loc[:, expr]]
            _names = [expr]
        else:  # pragma: no cover
            msg = f"Expected expression or column name, got: {expr}"
            raise TypeError(msg)

        results.update(dict(zip(_names, _results)))

    for name, value in named_exprs.items():
        _results = value._call(df)
        if len(_results) != 1:  # pragma: no cover
            msg = "Named expressions must return a single column"
            raise AssertionError(msg)
        results[name] = _results[0]

    return results


def parse_series(named_series: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    de = get_dask_expr()
    series = list(named_series.values())
    lengths = [
        1 if isinstance(s, de._collection.Scalar) else len(s.index) for s in series
    ]
    max_length = max(lengths)

    idx = series[lengths.index(max_length)].index
    parsed_series = {
        name: value
        if length > 1
        else value.compute()
        if isinstance(value, de._collection.Scalar)
        else value.loc[0][0]
        for (name, value), length in zip(named_series.items(), lengths)
    }

    return idx, parsed_series
