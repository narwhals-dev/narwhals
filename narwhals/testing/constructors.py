from __future__ import annotations

import sys
from importlib.util import find_spec

from narwhals._utils import Implementation

if not ("pytest" in sys.modules or find_spec("pytest")):
    msg = (
        "narwhals.testing.constructors requires the 'pytest' module\n"
        "Please install it using the command: pip install pytest"
    )
    raise ModuleNotFoundError(msg)


from typing import TYPE_CHECKING

import pytest

from narwhals.testing._utils import (
    MIN_PANDAS_NULLABLE_VERSION,
    PANDAS_VERSION,
    PYARROW_AVAILABLE,
    backend_is_available,
    get_constructors,
)

if TYPE_CHECKING:
    from narwhals.testing.typing import Constructor, ConstructorEager, ConstructorLazy

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
    """Pytest fixture that returns all eager dataframe constructors supported by Narwhals and are installed.

    Notes:
        This function is intended to be used in unit tests and requires pytest to be installed.

    Examples:
        >>> import narwhals as nw
        >>> from narwhals.testing.constructors import eager_constructor
        >>> from narwhals.testing.typing import ConstructorEager
        >>>
        >>> def test_shape(eager_constructor: ConstructorEager) -> None:
        ...     data = {"x": [1, 2, 3], "y": [7.1, 8.2, 9.3]}
        ...     native_frame = eager_constructor(data)
        ...     nw_frame = nw.from_native(native_frame, eager_only=True)
        ...     assert nw_frame.shape == (3, 2)
    """
    return request.param  # type: ignore[no-any-return]


@pytest.fixture(params=lazy_constructors.copy(), ids=lazy_ids.copy())
def lazy_constructor(request: pytest.FixtureRequest) -> ConstructorLazy:
    """Pytest fixture that returns all lazy dataframe constructors supported by Narwhals and are installed.

    Notes:
        This function is intended to be used in unit tests and requires pytest to be installed.

    Examples:
        >>> import narwhals as nw
        >>> from narwhals.testing.constructors import lazy_constructor
        >>> from narwhals.testing.typing import ConstructorLazy
        >>>
        >>> def test_schema(lazy_constructor: ConstructorLazy) -> None:
        ...     data = {"x": [1, 2, 3], "y": [7.1, 8.2, 9.3]}
        ...     native_frame = lazy_constructor(data)
        ...     nw_frame = nw.from_native(native_frame)
        ...     assert nw_frame.collect_schema() == {"x": nw.Int64(), "y": nw.Float64()}
    """
    return request.param  # type: ignore[no-any-return]


@pytest.fixture(
    params=[*eager_constructors, *lazy_constructors], ids=[*eager_ids, *lazy_ids]
)
def frame_constructor(request: pytest.FixtureRequest) -> Constructor:
    """Pytest fixture that returns all eager and lazy dataframe constructors supported by Narwhals and are installed.

    Notes:
        This function is intended to be used in unit tests and requires pytest to be installed.

    Examples:
        >>> import narwhals as nw
        >>> from narwhals.testing.constructors import frame_constructor
        >>> from narwhals.testing.typing import Constructor
        >>>
        >>> def test_schema(frame_constructor: Constructor) -> None:
        ...     data = {"x": [1, 2, 3], "y": [7.1, 8.2, 9.3]}
        ...     native_frame = frame_constructor(data)
        ...     nw_frame = nw.from_native(native_frame)
        ...     assert nw_frame.collect_schema() == {"x": nw.Int64(), "y": nw.Float64()}
    """
    return request.param  # type: ignore[no-any-return]
