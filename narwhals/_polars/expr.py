from narwhals._polars.utils import extract_args_kwargs


class PolarsExpr:
    def __init__(self, expr):
        self._native_expr = expr

    def __repr__(self):
        return "PolarsExpr"

    def __narwhals_expr__(self):
        return self

    # def __narwhals_namespace__(self):
    #     return PolarsNamespace()

    def _from_native_expr(self, expr):
        return self.__class__(expr)

    def __getattr__(self, attr):
        def func(*args, **kwargs):
            args, kwargs = extract_args_kwargs(args, kwargs)
            return self._from_native_expr(
                getattr(self._native_expr, attr)(*args, **kwargs)
            )

        return func
