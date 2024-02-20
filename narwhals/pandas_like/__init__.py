from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from narwhals.pandas_like.dataframe import LazyFrame

if TYPE_CHECKING:
    from narwhals.spec import LazyFrame as LazyFrameT
    from narwhals.spec import Namespace as NamespaceT


def translate(
    df: Any,
    implementation: str,
    api_version: str,
) -> tuple[LazyFrameT, NamespaceT]:
    df = LazyFrame(
        df,
        api_version=api_version,
        implementation=implementation,
    )
    return df, df.__lazyframe_namespace__()
