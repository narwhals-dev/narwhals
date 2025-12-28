"""Flags for features not available in all supported `pyarrow` versions."""

from __future__ import annotations

from typing import Final

from narwhals._utils import Implementation

BACKEND_VERSION = Implementation.PYARROW._backend_version()
"""Static backend version for `pyarrow`."""

RANK_ACCEPTS_CHUNKED: Final = BACKEND_VERSION >= (14,)

HAS_FROM_TO_STRUCT_ARRAY: Final = BACKEND_VERSION >= (15,)
"""`pyarrow.Table.{from,to}_struct_array` added in https://github.com/apache/arrow/pull/38520"""


TABLE_RENAME_ACCEPTS_DICT: Final = BACKEND_VERSION >= (17,)

TAKE_ACCEPTS_TUPLE: Final = BACKEND_VERSION >= (18,)

HAS_STRUCT_TYPE_FIELDS: Final = BACKEND_VERSION >= (18,)
"""`pyarrow.StructType.{fields,names}` added in https://github.com/apache/arrow/pull/43481"""

HAS_SCATTER: Final = BACKEND_VERSION >= (20,)
"""`pyarrow.compute.scatter` added in https://github.com/apache/arrow/pull/44394"""

HAS_KURTOSIS_SKEW = BACKEND_VERSION >= (20,)
"""`pyarrow.compute.{kurtosis,skew}` added in https://github.com/apache/arrow/pull/45677"""

HAS_PIVOT_WIDER = BACKEND_VERSION >= (20,)
"""`pyarrow.compute.pivot_wider` added in https://github.com/apache/arrow/pull/45562"""

HAS_ARANGE: Final = BACKEND_VERSION >= (21,)
"""`pyarrow.arange` added in https://github.com/apache/arrow/pull/46778"""

TO_STRUCT_ARRAY_ACCEPTS_EMPTY: Final = BACKEND_VERSION >= (21,)
"""`pyarrow.Table.to_struct_array` fixed in https://github.com/apache/arrow/pull/46357"""

HAS_ZFILL: Final = BACKEND_VERSION >= (21,)
"""`pyarrow.compute.utf8_zero_fill` added in https://github.com/apache/arrow/pull/46815"""
