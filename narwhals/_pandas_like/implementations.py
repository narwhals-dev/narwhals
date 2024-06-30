from narwhals.dependencies import Backend, get_implementation
from typing import Any, Literal

from typing_extensions import assert_never

from functools import partial

PANDAS_IMPLEMENTATIONS = Literal[Backend.PANDAS, Backend.MODIN, Backend.CUDF]


def get_series_implementation(backend: PANDAS_IMPLEMENTATIONS) -> Any:
    implementation = get_implementation(backend)
    
    if backend is Backend.PANDAS:
        return partial(implementation.Series, copy=False)
    if backend is Backend.MODIN or backend is Backend.CUDF:
        return implementation.Series
    
    return assert_never(backend)
