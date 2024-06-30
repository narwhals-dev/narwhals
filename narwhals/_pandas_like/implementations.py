from functools import partial
from typing import Any
from typing import Literal

from typing_extensions import assert_never

from narwhals.dependencies import Backend
from narwhals.dependencies import get_implementation

PANDAS_IMPLEMENTATIONS = Literal[Backend.PANDAS, Backend.MODIN, Backend.CUDF]


def get_series_implementation(backend: PANDAS_IMPLEMENTATIONS) -> Any:
    implementation = get_implementation(backend)

    if backend is Backend.PANDAS:
        return partial(implementation.Series, copy=False)
    if backend is Backend.MODIN or backend is Backend.CUDF:
        return implementation.Series

    return assert_never(backend)
