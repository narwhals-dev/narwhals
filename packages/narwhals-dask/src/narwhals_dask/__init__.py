"""Dask backend for Narwhals.

Once installed, Narwhals discovers this plugin through the `narwhals.plugins` entry point
and uses it for all `dask.dataframe.DataFrame` objects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import dask.dataframe as dd
    from typing_extensions import TypeIs

    from narwhals.utils import Version
    from narwhals_dask.namespace import DaskNamespace

__all__ = ["NATIVE_PACKAGE", "__narwhals_namespace__", "is_native"]

NATIVE_PACKAGE = "dask"


def __narwhals_namespace__(version: Version) -> DaskNamespace:  # noqa: N807
    from narwhals_dask.namespace import DaskNamespace

    return DaskNamespace(version=version)


def is_native(native_object: object) -> TypeIs[dd.DataFrame]:
    import dask.dataframe as dd

    return isinstance(native_object, dd.DataFrame)
