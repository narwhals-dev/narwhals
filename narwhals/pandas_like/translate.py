from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import overload

if TYPE_CHECKING:
    from narwhals.pandas_like.dataframe import DataFrame
    from narwhals.pandas_like.dataframe import LazyFrame
    from narwhals.pandas_like.namespace import Namespace


@overload
def translate(
    df: Any,
    implementation: str,
    api_version: str,
    *,
    eager: Literal[True],
) -> tuple[DataFrame, Namespace]:
    ...


@overload
def translate(
    df: Any,
    implementation: str,
    api_version: str,
    *,
    eager: Literal[False],
) -> tuple[LazyFrame, Namespace]:
    ...


@overload
def translate(
    df: Any,
    implementation: str,
    api_version: str,
    *,
    eager: bool,
) -> tuple[DataFrame | LazyFrame, Namespace]:
    ...


def translate(
    df: Any,
    implementation: str,
    api_version: str,
    *,
    eager: bool,
) -> tuple[LazyFrame | DataFrame, Namespace]:
    from narwhals.pandas_like.dataframe import DataFrame
    from narwhals.pandas_like.dataframe import LazyFrame
    from narwhals.pandas_like.utils import get_namespace

    if eager:
        df = DataFrame(
            df,
            api_version=api_version,
            implementation=implementation,
        )
    else:
        df = LazyFrame(
            df,
            api_version=api_version,
            implementation=implementation,
        )
    return df, get_namespace(df)
