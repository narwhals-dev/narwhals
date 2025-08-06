from __future__ import annotations

import sys
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any, Callable

from narwhals._utils import Implementation

if not ("pytest" in sys.modules or find_spec("pytest")):
    msg = (
        "narwhals.testing.constructors requires the 'pytest' module\n"
        "Please install it using the command: pip install pytest"
    )
    raise ModuleNotFoundError(msg)


import pytest

from narwhals.testing._constructors import (
    MIN_PANDAS_NULLABLE_VERSION,
    PANDAS_VERSION,
    PYARROW_AVAILABLE,
    backend_is_available,
    get_constructors,
)

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

    from narwhals.typing import DataFrameLike, NativeFrame, NativeLazyFrame

    Data: TypeAlias = "dict[str, Any]"

    Constructor: TypeAlias = Callable[
        [Any], "NativeLazyFrame | NativeFrame | DataFrameLike"
    ]
    ConstructorEager: TypeAlias = Callable[[Any], "NativeFrame | DataFrameLike"]
    ConstructorLazy: TypeAlias = Callable[[Any], "NativeLazyFrame"]


selected_constructors: list[str] = []

for impl in Implementation:
    if impl in {
        Implementation.UNKNOWN,
        Implementation.PYSPARK_CONNECT,
    } or not backend_is_available(impl=impl):
        continue

    if impl.is_polars():
        selected_constructors.extend(("polars[eager]", "polars[lazy]"))
    elif impl.is_pandas():
        selected_constructors.append("pandas")
        if PANDAS_VERSION >= MIN_PANDAS_NULLABLE_VERSION:
            selected_constructors.append("pandas[nullable]")
            if PYARROW_AVAILABLE:
                selected_constructors.append("pandas[pyarrow]")
    elif impl.is_modin():
        selected_constructors.append("modin")
        if PANDAS_VERSION >= MIN_PANDAS_NULLABLE_VERSION and PYARROW_AVAILABLE:
            selected_constructors.append("modin[pyarrow]")
    else:
        selected_constructors.append(impl.value)


eager_constructors, eager_ids, lazy_constructors, lazy_ids = get_constructors(
    selected_constructors
)


@pytest.fixture(params=eager_constructors.copy(), ids=eager_ids.copy())
def eager_constructor(request: pytest.FixtureRequest) -> ConstructorEager:
    """Provides a (eager) dataframe constructor configured for testing, based on installed libraries."""
    return request.param  # type: ignore[no-any-return]


@pytest.fixture(params=lazy_constructors.copy(), ids=lazy_ids.copy())
def lazy_constructor(request: pytest.FixtureRequest) -> Constructor:
    """Provides a lazyframe constructor configured for testing, based on installed libraries."""
    return request.param  # type: ignore[no-any-return]


@pytest.fixture(
    params=[*eager_constructors, *lazy_constructors], ids=[*eager_ids, *lazy_ids]
)
def frame_constructor(request: pytest.FixtureRequest) -> Constructor:
    """Provides a eager or lazy frame constructor configured for testing, based on installed libraries."""
    return request.param  # type: ignore[no-any-return]
