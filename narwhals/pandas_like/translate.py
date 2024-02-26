from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from narwhals.pandas_like.dataframe import DataFrame
    from narwhals.pandas_like.namespace import Namespace
    from narwhals.pandas_like.series import Series


def translate_frame(
    df: Any,
    implementation: str,
    *,
    is_eager: bool,
    is_lazy: bool,
) -> tuple[DataFrame, Namespace]:
    from narwhals.pandas_like.dataframe import DataFrame
    from narwhals.pandas_like.namespace import Namespace

    df = DataFrame(
        df,
        implementation=implementation,
        is_eager=is_eager,
        is_lazy=is_lazy,
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
