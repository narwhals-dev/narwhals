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
        if obj._returns_scalar:
            # Return scalar, let Dask do its broadcasting
            return result[0]
        return result
    return obj


def parse_exprs_and_named_exprs(
    df: DaskLazyFrame, *exprs: Any, **named_exprs: Any
) -> dict[str, Any]:
    results = {}
    for expr in exprs:
        if hasattr(expr, "__narwhals_expr__"):
            _results = expr._call(df)
        elif isinstance(expr, str):
            _results = [df._native_dataframe.loc[:, expr]]
        else:  # pragma: no cover
            msg = f"Expected expression or column name, got: {expr}"
            raise TypeError(msg)
        for _result in _results:
            if getattr(expr, "_returns_scalar", False):
                results[_result.name] = _result[0]
            else:
                results[_result.name] = _result
    for name, value in named_exprs.items():
        _results = value._call(df)
        if len(_results) != 1:  # pragma: no cover
            msg = "Named expressions must return a single column"
            raise AssertionError(msg)
        for _result in _results:
            if getattr(value, "_returns_scalar", False):
                results[name] = _result[0]
            else:
                results[name] = _result
    return results


def add_row_index(frame: Any, name: str) -> Any:
    frame = frame.assign(**{name: 1})
    return frame.assign(**{name: frame[name].cumsum(method="blelloch") - 1})
