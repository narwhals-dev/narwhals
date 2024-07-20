import polars as pl

from narwhals._polars.utils import extract_args_kwargs


class PolarsNamespace:
    def _from_native_expr(self, expr):
        from narwhals._polars.expr import PolarsExpr

        return PolarsExpr(expr)

    def __getattr__(self, attr):
        def func(*args, **kwargs):
            args, kwargs = extract_args_kwargs(args, kwargs)
            return self._from_native_expr(getattr(pl, attr)(*args, **kwargs))

        return func
