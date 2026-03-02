from __future__ import annotations

from typing import TYPE_CHECKING, Any, overload

from narwhals._typing_compat import deprecated
from narwhals._utils import Implementation, Version, is_eager_allowed

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

    from narwhals._plan import arrow as _arrow
    from narwhals._plan.compliant.dataframe import EagerDataFrame
    from narwhals._plan.compliant.namespace import CompliantNamespace, EagerNamespace
    from narwhals._plan.compliant.typing import NamespaceT_co, SupportsNarwhalsNamespace
    from narwhals._plan.plans.visitors import ResolvedToCompliant
    from narwhals._plan.typing import NativeDataFrameT, NativeSeriesT
    from narwhals._typing import Arrow, _EagerAllowedImpl, _LazyAllowedImpl
    from narwhals.typing import Backend, EagerAllowed, IntoBackend

    # NOTE: Use when you have a function that calls a namespace method, and eventually returns:
    # - `DataFrame[NativeDataFrameT]`, or
    # - `Series[NativeSeriesT]`
    EagerNs: TypeAlias = EagerNamespace[
        EagerDataFrame[Any, NativeDataFrameT, Any], Any, Any, Any, NativeSeriesT
    ]

KnownImpl: TypeAlias = "_EagerAllowedImpl | _LazyAllowedImpl"
"""Equivalent to `Backend - BackendName`."""


def namespace(obj: SupportsNarwhalsNamespace[NamespaceT_co], /) -> NamespaceT_co:
    """Get the compliant namespace from `obj`."""
    return obj.__narwhals_namespace__()


def namespace_from_backend(
    backend: IntoBackend[Backend],
) -> CompliantNamespace[Any, Any, Any]:
    """Instantiate a compliant namespace from `backend`, routing through `Implementation`."""
    impl = Implementation.from_backend(backend)
    if impl is Implementation.POLARS:
        from narwhals._plan import polars as _polars

        return _polars.Namespace(Version.MAIN)
    if impl is Implementation.PYARROW:
        from narwhals._plan import arrow as _arrow

        return _arrow.Namespace(Version.MAIN)
    msg = f"Not yet supported in `narwhals._plan`, got: {impl!r}"
    raise NotImplementedError(msg)


# TODO @dangotbanned: Use the more granular protocols for `DataFrame.from_dict`
@overload
def eager_namespace_from_backend(backend: Arrow, /) -> _arrow.Namespace: ...
@overload
def eager_namespace_from_backend(
    backend: IntoBackend[EagerAllowed], /
) -> EagerNs[Any, Any]: ...
@deprecated("Use the more granualar protocols instead")
def eager_namespace_from_backend(
    backend: IntoBackend[EagerAllowed], /
) -> EagerNs[Any, Any] | _arrow.Namespace:
    impl = eager_implementation(backend)
    if impl is Implementation.PYARROW:
        from narwhals._plan import arrow as _arrow

        return _arrow.Namespace(Version.MAIN)

    raise NotImplementedError(impl)


# TODO @dangotbanned: Need to be able to store a closure for getting namespaces
def known_implementation(backend: IntoBackend[Backend] | Any) -> KnownImpl:
    """Reject the possibility of plugins via this path."""
    impl = Implementation.from_backend(backend)
    if impl is Implementation.UNKNOWN:
        msg = f"{impl!r} is not supported in this context, got:\n{backend!r}"
        raise NotImplementedError(msg)
    return impl


def eager_implementation(backend: IntoBackend[Backend] | Any) -> _EagerAllowedImpl:
    impl = Implementation.from_backend(backend)
    if is_eager_allowed(impl):
        return impl
    msg = f"{impl} support in Narwhals is lazy-only"
    raise ValueError(msg)


def evaluator(backend: KnownImpl) -> type[ResolvedToCompliant[Any]]:
    if backend is Implementation.POLARS:
        from narwhals._plan.polars.lazyframe import PolarsEvaluator

        return PolarsEvaluator
    raise NotImplementedError(backend)
