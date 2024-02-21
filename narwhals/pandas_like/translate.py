from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from narwhals.pandas_like.dataframe import LazyFrame
    from narwhals.pandas_like.namespace import Namespace


def translate(
    df: Any,
    implementation: str,
    api_version: str,
) -> tuple[LazyFrame, Namespace]:
    from narwhals.pandas_like.dataframe import LazyFrame

    df = LazyFrame(
        df,
        api_version=api_version,
        implementation=implementation,
    )
    return df, df.__lazyframe_namespace__()
