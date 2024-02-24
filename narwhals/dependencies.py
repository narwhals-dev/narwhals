import functools
from typing import Any


@functools.lru_cache
def get_polars() -> Any:
    try:
        import polars
    except ImportError:
        return None
    return polars


@functools.lru_cache
def get_pandas() -> Any:
    try:
        import pandas
    except ImportError:
        return None
    return pandas


@functools.lru_cache
def get_modin() -> Any:
    try:
        import modin.pandas as mpd
    except ImportError:
        return None
    return mpd


@functools.lru_cache
def get_cudf() -> Any:
    try:
        import cudf
    except ImportError:
        return None
    return cudf
