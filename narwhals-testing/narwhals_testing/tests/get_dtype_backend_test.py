from __future__ import annotations

from contextlib import suppress
from importlib.util import find_spec
from itertools import chain
from typing import TYPE_CHECKING, TypeVar

import pytest

from narwhals._utils import Implementation

pytest.importorskip("pandas")
from pandas.api.extensions import ExtensionDtype

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
) -> Iterator[ExtensionDtype]:
    """Skips dtypes that are not compatible on the tested version of pandas."""
    for dtype in dtypes:  # pragma: no cover
        # NOTE: Slightly different errors between implementations
        # TypeError: https://github.com/pandas-dev/pandas/blob/2cc37625532045f4ac55b27176454bbbc9baf213/pandas/core/dtypes/base.py#L261-L264
        # NotImplementedError: https://github.com/rapidsai/cudf/blob/22d1b7c75e892f9136e11e47c09e7913910414ea/python/cudf/cudf/core/dtypes.py#L318-L319
        with suppress(TypeError, NotImplementedError):
            result = dtype.construct_from_string(alias)
            yield result
            return


def generate_pandas_dtypes_public(aliases: Iterable[str]) -> list[ExtensionDtype]:
    """Technically only uses the public api."""
    dtype_classes = _deep_descendants(ExtensionDtype)
    it = chain.from_iterable(
        brute_force_construct_dtype(alias, dtype_classes) for alias in aliases
    )
    return list(it)


@pytest.mark.parametrize(
    "implementation", [Implementation.PANDAS, Implementation.CUDF, Implementation.MODIN]
)
@pytest.mark.parametrize(
    "pandas_dtype", generate_pandas_dtypes_public(SIMPLE_DTYPE_ALIASES), ids=str
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
