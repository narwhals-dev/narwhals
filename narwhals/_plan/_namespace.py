# TODO @dangotbanned: Review what to extract into plugin features
from __future__ import annotations

from typing import TYPE_CHECKING, Any, get_args

from narwhals._typing import _LazyFrameCollectImpl
from narwhals._utils import (
    Implementation,
    Version,
    can_lazyframe_collect,
    is_eager_allowed,
)

if TYPE_CHECKING:
    import polars as pl
    from typing_extensions import TypeAlias

    from narwhals._plan.compliant import typing as ct
    from narwhals._plan.compliant.package import HasPlanEvaluator
    from narwhals._plan.plans.visitors import ResolvedToCompliant
    from narwhals._typing import _EagerAllowedImpl, _LazyAllowedImpl
    from narwhals.typing import Backend, IntoBackend


KnownImpl: TypeAlias = "_EagerAllowedImpl | _LazyAllowedImpl"
"""Equivalent to `Backend - BackendName`."""


def namespace(obj: ct.SupportsNarwhalsNamespace[ct.NamespaceT_co], /) -> ct.NamespaceT_co:
    """Get the compliant namespace from `obj`."""
    return obj.__narwhals_namespace__()


def namespace_from_backend(backend: IntoBackend[Backend] | Any) -> ct.NamespaceAny:
    """Instantiate a compliant namespace from `backend`, routing through `Implementation`."""
    impl = Implementation.from_backend(backend)
    if impl is Implementation.POLARS:
        from narwhals._plan import polars as _polars

        return _polars.Namespace()
    if impl is Implementation.PYARROW:
        from narwhals._plan import arrow as _arrow

        return _arrow.Namespace()
    msg = f"Not yet supported in `narwhals._plan`, got: {impl!r}"
    raise NotImplementedError(msg)


# TODO @dangotbanned: Need to be able to store a closure for getting namespaces
def known_implementation(backend: IntoBackend[Backend] | Any) -> KnownImpl:
    """Reject the possibility of plugins via this path."""
    impl = Implementation.from_backend(backend)
    if impl is Implementation.UNKNOWN:
        msg = f"{impl!r} is not supported in this context, got: {backend!r}"
        raise NotImplementedError(msg)
    return impl


def eager_implementation(backend: IntoBackend[Backend] | Any) -> _EagerAllowedImpl:
    impl = known_implementation(backend)
    if is_eager_allowed(impl):
        return impl
    msg = f"{impl} support in Narwhals is lazy-only"  # pragma: no cover
    raise TypeError(msg)  # pragma: no cover


def collect_implementation(backend: IntoBackend[Backend] | Any) -> _LazyFrameCollectImpl:
    """Parse `backend` into an `Implementation`, ensuring it can be used in `LazyFrame.collect`."""
    impl = Implementation.from_backend(backend)
    if can_lazyframe_collect(impl):
        return impl
    msg = (  # pragma: no cover
        f"Unsupported `backend` value.\n"
        f"Expected one of {get_args(_LazyFrameCollectImpl)} or None, got: {impl}."
    )
    raise TypeError(msg)  # pragma: no cover


def evaluator(backend: KnownImpl, version: Version) -> type[ResolvedToCompliant[Any]]:
    if backend is Implementation.POLARS:
        from narwhals._plan import polars as _polars

        pl_module: HasPlanEvaluator[pl.LazyFrame]
        if version is Version.MAIN:
            pl_module = _polars
        elif version is Version.V1:  # pragma: no cover
            pl_module = _polars.v1
        else:
            raise NotImplementedError(version)
        result: type[ResolvedToCompliant[Any]] = pl_module.PlanEvaluator
        return result
    raise NotImplementedError(backend)
