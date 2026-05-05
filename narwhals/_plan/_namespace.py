from __future__ import annotations

from typing import TYPE_CHECKING, Any

from narwhals._utils import Implementation

if TYPE_CHECKING:
    from narwhals._plan.compliant import typing as ct
    from narwhals._plan.typing import KnownImpl
    from narwhals.typing import Backend, IntoBackend


def namespace(obj: ct.SupportsNarwhalsNamespace[ct.NamespaceT_co], /) -> ct.NamespaceT_co:
    """Get the compliant namespace from `obj`."""
    return obj.__narwhals_namespace__()


# TODO @dangotbanned: (after everything else) Review how functions from namespace should work
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


# TODO @dangotbanned: Replace with `PluginManager.(builtin|builtin_name)`?
def known_implementation(backend: IntoBackend[Backend] | Any) -> KnownImpl:
    """Reject the possibility of plugins via this path."""
    impl = Implementation.from_backend(backend)
    if impl is Implementation.UNKNOWN:
        msg = f"{impl!r} is not supported in this context, got: {backend!r}"
        raise NotImplementedError(msg)
    return impl
