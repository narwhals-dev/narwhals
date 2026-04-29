from __future__ import annotations

import typing as _t

from narwhals._plan.compliant import plugins as _plugins

if _t.TYPE_CHECKING:
    from collections.abc import Iterator

    import polars as pl
    from typing_extensions import TypeIs

    from narwhals._native import NativePolars
    from narwhals._plan.polars import v1, v2
    from narwhals._plan.polars.classes import PolarsClasses
    from narwhals._plan.polars.dataframe import PolarsDataFrame as DataFrame
    from narwhals._plan.polars.expr import PolarsExpr as Expr, PolarsExpr as Scalar
    from narwhals._plan.polars.lazyframe import (
        PolarsEvaluator as PlanEvaluator,
        PolarsLazyFrame as LazyFrame,
    )
    from narwhals._plan.polars.namespace import PolarsNamespace as Namespace
    from narwhals._plan.polars.series import PolarsSeries as Series

PlanResolver = None

__all__ = [
    "DataFrame",
    "Expr",
    "LazyFrame",
    "Namespace",
    "PlanEvaluator",
    "PlanResolver",
    "PolarsPlugin",
    "Scalar",
    "Series",
    "plugin",
    "v1",
    "v2",
]


@_t.final
class PolarsPlugin(
    _plugins.Builtin["PolarsClasses", "pl.DataFrame", "pl.LazyFrame", "pl.Series"]
):
    __slots__ = ()
    implementation = _plugins.Implementation.POLARS
    requirements = ("polars",)

    def is_native(self, obj: _t.Any) -> TypeIs[NativePolars]:
        import polars as pl

        return isinstance(obj, (pl.DataFrame, pl.LazyFrame, pl.Series))

    def native_dataframe_classes(self) -> Iterator[type[pl.DataFrame]]:
        import polars as pl

        yield pl.DataFrame

    def native_lazyframe_classes(self) -> Iterator[type[pl.LazyFrame]]:
        import polars as pl

        yield pl.LazyFrame

    def native_series_classes(self) -> Iterator[type[pl.Series]]:
        import polars as pl

        yield pl.Series

    @property
    def __narwhals_classes__(self) -> PolarsClasses:
        from narwhals._plan.polars.classes import PolarsClasses

        return PolarsClasses()


plugin = PolarsPlugin()


if not _t.TYPE_CHECKING:

    def __getattr__(name: str) -> _t.Any:
        from importlib import import_module

        if name not in __all__:
            msg = f"module {__name__!r} has no attribute {name!r}"
            raise AttributeError(msg)

        package_name = "narwhals._plan.polars"
        prefix = "Polars"
        # NOTE: Yes, this is quite ugly - just need something to keep these imports working temporarily
        globs = globals()
        if name in {"v1", "v2"}:
            vn = globs[name] = import_module(f"{package_name}.{name}")
            return vn
        if name in {"Scalar", "Expr"}:
            expr = import_module(f"{package_name}.expr")
            globs.update(Expr=expr.PolarsExpr, Scalar=expr.PolarsExpr)
            return globs[name]
        if name in {"LazyFrame", "PlanEvaluator"}:
            lazyframe = import_module(f"{package_name}.lazyframe")
            globs.update(
                LazyFrame=lazyframe.PolarsLazyFrame,
                PlanEvaluator=lazyframe.PolarsEvaluator,
            )
            return globs[name]
        # "DataFrame", "Namespace", "Series"
        module = import_module(f"{package_name}.{name.lower()}")
        tp = globs[name] = getattr(module, f"{prefix}{name}")
        return tp
