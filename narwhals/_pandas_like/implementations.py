from functools import partial
from typing import Any
from typing import Literal

from typing_extensions import assert_never

from narwhals.dependencies import Implementation
from narwhals.dependencies import get_backend

PANDAS_IMPLEMENTATIONS = Literal[Implementation.PANDAS, Implementation.MODIN, Implementation.CUDF]


def get_series_implementation(implementation: PANDAS_IMPLEMENTATIONS) -> Any:
    implementation = get_backend(implementation)

    if implementation is Implementation.PANDAS:
        return partial(implementation.Series, copy=False)
    if implementation is Implementation.MODIN or implementation is Implementation.CUDF:
        return implementation.Series

    return assert_never(implementation)
