from __future__ import annotations

import typing as _t

from narwhals._plan.compliant import plugins as _plugins

if _t.TYPE_CHECKING:
    from collections.abc import Iterator

    import pyarrow as pa
    from typing_extensions import TypeIs

    from narwhals._native import NativeArrow
    from narwhals._plan.arrow import v1, v2
    from narwhals._plan.arrow.classes import ArrowClasses
    from narwhals._plan.arrow.dataframe import ArrowDataFrame as DataFrame
    from narwhals._plan.arrow.expr import ArrowExpr as Expr, ArrowScalar as Scalar
    from narwhals._plan.arrow.lazyframe import ArrowLazyFrame as LazyFrame
    from narwhals._plan.arrow.namespace import ArrowNamespace as Namespace
    from narwhals._plan.arrow.series import ArrowSeries as Series


PlanResolver = None
PlanEvaluator = None

__all__ = (
    "ArrowPlugin",
    "DataFrame",
    "Expr",
    "LazyFrame",
    "Namespace",
    "PlanEvaluator",
    "PlanResolver",
    "Scalar",
    "Series",
    "plugin",
    "v1",
    "v2",
)


@_t.final
class ArrowPlugin(
    _plugins.Builtin["ArrowClasses", "pa.Table", _t.Any, "pa.ChunkedArray[_t.Any]"]
):
    __slots__ = ()
    implementation = _plugins.Implementation.PYARROW
    requirements = ("pyarrow",)

    def is_native(self, obj: _t.Any) -> TypeIs[NativeArrow]:
        import pyarrow as pa

        return isinstance(obj, (pa.Table, pa.ChunkedArray))

    def native_dataframe_classes(self) -> Iterator[type[pa.Table]]:
        import pyarrow as pa

        yield pa.Table

    def native_series_classes(self) -> Iterator[type[pa.ChunkedArray[_t.Any]]]:
        import pyarrow as pa

        yield pa.ChunkedArray

    def native_lazyframe_classes(self) -> Iterator[type[_t.Any]]:
        yield from ()

    @property
    def __narwhals_classes__(self) -> ArrowClasses:
        from narwhals._plan.arrow.classes import ArrowClasses

        return ArrowClasses()


plugin = ArrowPlugin()


if not _t.TYPE_CHECKING:

    def __getattr__(name: str) -> _t.Any:
        from importlib import import_module

        if name not in __all__:
            msg = f"module {__name__!r} has no attribute {name!r}"
            raise AttributeError(msg)

        package_name = "narwhals._plan.arrow"
        prefix = "Arrow"
        # NOTE: Yes, this is quite ugly - just need something to keep these imports working temporarily
        globs = globals()
        if name in {"v1", "v2"}:
            vn = globs[name] = import_module(f"{package_name}.{name}")
            return vn
        if name in {"Scalar", "Expr"}:
            expr = import_module(f"{package_name}.expr")
            globs.update(Expr=expr.ArrowExpr, Scalar=expr.ArrowScalar)
            return globs[name]
        # "DataFrame", "LazyFrame", "Namespace", "Series"
        module = import_module(f"{package_name}.{name.lower()}")
        tp = globs[name] = getattr(module, f"{prefix}{name}")
        return tp
