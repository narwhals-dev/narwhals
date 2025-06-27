from __future__ import annotations

from contextlib import suppress
from importlib.util import find_spec
from typing import TYPE_CHECKING, TypeVar

import numpy as np
import pytest
from pandas.api.extensions import ExtensionDtype

from narwhals._utils import Implementation

pytest.importorskip("pandas")
import pandas as pd

from narwhals._pandas_like.utils import get_dtype_backend

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

T = TypeVar("T")

PYARROW_ALIASES = (
    "Int64[pyarrow]",
    "int64[pyarrow]",
    "Int32[pyarrow]",
    "int32[pyarrow]",
    "Int16[pyarrow]",
    "int16[pyarrow]",
    "Int8[pyarrow]",
    "int8[pyarrow]",
    "UInt64[pyarrow]",
    "uint64[pyarrow]",
    "UInt32[pyarrow]",
    "uint32[pyarrow]",
    "UInt16[pyarrow]",
    "uint16[pyarrow]",
    "UInt8[pyarrow]",
    "uint8[pyarrow]",
    "Float64[pyarrow]",
    "float64[pyarrow]",
    "double[pyarrow]",
    "Float32[pyarrow]",
    "float32[pyarrow]",
    "float[pyarrow]",
    "string[pyarrow]",
    "large_string[pyarrow]",
    "boolean[pyarrow]",
    "bool[pyarrow]",
    "date32[day][pyarrow]",
    "date32[pyarrow]",
    "duration[ns][pyarrow]",
    "duration[us][pyarrow]",
    "duration[ms][pyarrow]",
    "duration[s][pyarrow]",
)
NUMPY_NULLABLE_ALIASES = [
    "Int64",
    "Int32",
    "Int16",
    "Int8",
    "UInt64",
    "UInt32",
    "UInt16",
    "UInt8",
    "Float64",
    "Float32",
    "boolean",
]
PANDAS_NON_NULLABLE_ALIASES = [
    "object",
    "int64",
    "int32",
    "int16",
    "int8",
    "uint64",
    "uint32",
    "uint16",
    "uint8",
    "float64",
    "float32",
    "string",
    "string[python]",
    "bool",
    "category",
    "timedelta64[ns]",
    "timedelta64[us]",
    "timedelta64[ms]",
    "timedelta64[s]",
]


SIMPLE_DTYPE_ALIASES = (
    (*PANDAS_NON_NULLABLE_ALIASES, *NUMPY_NULLABLE_ALIASES, *PYARROW_ALIASES)
    if find_spec("pyarrow")
    else (*PANDAS_NON_NULLABLE_ALIASES, *NUMPY_NULLABLE_ALIASES)
)
"""Excluding timezones, dictionary, maybe others?"""


def _iter_leaves(root: type[T]) -> Iterator[type[T]]:
    for child in root.__subclasses__():
        if child.__subclasses__():
            yield from _iter_leaves(child)
        else:
            yield child


def _deep_descendants(root: type[T]) -> list[type[T]]:
    """Return all subclasses of `root` that have no subclasses.

    Note:
        For pandas, this gives us a public *almost* equvialent to `pandas.core.dtypes.base._registry.dtypes`.
    """
    return sorted(set(_iter_leaves(root)), key=repr)


def brute_force_construct_dtype(
    alias: str, dtypes: Iterable[type[ExtensionDtype]]
) -> ExtensionDtype:
    for dtype in dtypes:
        with suppress(TypeError):
            return dtype.construct_from_string(alias)
    msg = f"Found no constructor for {alias!r}"
    raise NotImplementedError(msg)


# NOTE: Option 1
def generate_pandas_dtypes_public(aliases: Iterable[str]) -> list[ExtensionDtype]:
    """Technically only uses the public api."""
    dtype_classes = _deep_descendants(ExtensionDtype)
    return [brute_force_construct_dtype(alias, dtype_classes) for alias in aliases]


def construct_dtype(alias: str) -> ExtensionDtype:
    dtype = pd.api.types.pandas_dtype(alias)
    if isinstance(dtype, np.dtype):
        return pd.core.dtypes.dtypes.NumpyEADtype(dtype)  # type: ignore[attr-defined, no-any-return]
    else:
        return dtype


# NOTE: Option 2
def generate_pandas_dtypes_mostly_public(aliases: Iterable[str]) -> list[ExtensionDtype]:
    """Ideally we'd use this, but `NumpyEADtype` is the ugly duckling."""
    return [construct_dtype(alias) for alias in aliases]


@pytest.fixture(
    params=generate_pandas_dtypes_public(SIMPLE_DTYPE_ALIASES), ids=SIMPLE_DTYPE_ALIASES
)
def pandas_dtype(request: pytest.FixtureRequest) -> ExtensionDtype:
    return request.param  # type: ignore[no-any-return]


@pytest.mark.parametrize(
    "implementation", [Implementation.PANDAS, Implementation.CUDF, Implementation.MODIN]
)
def test_get_dtype_backend(
    pandas_dtype: ExtensionDtype, implementation: Implementation
) -> None:
    result = get_dtype_backend(pandas_dtype, implementation)
    if implementation.is_cudf():
        assert result is None
    elif result == "pyarrow":
        assert pandas_dtype.name in PYARROW_ALIASES, pandas_dtype
    elif result is None:
        assert pandas_dtype.name in PANDAS_NON_NULLABLE_ALIASES, pandas_dtype
    else:
        assert pandas_dtype.name in NUMPY_NULLABLE_ALIASES, pandas_dtype
