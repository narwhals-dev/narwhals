from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import overload

if TYPE_CHECKING:
    from narwhals.pandas_like.dataframe import DataFrame
    from narwhals.pandas_like.dataframe import LazyFrame
    from narwhals.pandas_like.namespace import Namespace
    from narwhals.pandas_like.series import Series


@overload
def translate_frame(
    df: Any,
    implementation: str,
    *,
    eager_only: Literal[False],
    lazy_only: Literal[False],
) -> tuple[LazyFrame, Namespace]:
    ...


@overload
def translate_frame(
    df: Any,
    implementation: str,
    *,
    eager_only: Literal[False],
    lazy_only: Literal[True],
) -> tuple[LazyFrame, Namespace]:
    ...


@overload
def translate_frame(
    df: Any,
    implementation: str,
    *,
    eager_only: Literal[True],
    lazy_only: Literal[False],
) -> tuple[DataFrame, Namespace]:
    ...


@overload
def translate_frame(
    df: Any,
    implementation: str,
    *,
    eager_only: bool,
    lazy_only: bool,
) -> tuple[DataFrame | LazyFrame, Namespace]:
    ...


def translate_frame(
    df: Any,
    implementation: str,
    *,
    eager_only: bool,
    lazy_only: bool,
) -> tuple[LazyFrame | DataFrame, Namespace]:
    from narwhals.pandas_like.dataframe import DataFrame
    from narwhals.pandas_like.dataframe import LazyFrame
    from narwhals.pandas_like.namespace import Namespace

    if eager_only and not lazy_only:
        df = DataFrame(
            df,
            implementation=implementation,
        )
    else:
        df = LazyFrame(
            df,
            implementation=implementation,
        )
    return df, Namespace(implementation=implementation)


def translate_series(
    series: Any,
    implementation: str,
) -> tuple[Series, Namespace]:
    from narwhals.pandas_like.namespace import Namespace
    from narwhals.pandas_like.series import Series

    return Series(series, implementation=implementation), Namespace(
        implementation=implementation
    )
