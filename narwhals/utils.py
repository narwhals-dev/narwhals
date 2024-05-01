from __future__ import annotations

import re
from typing import Any
from typing import Iterable
from typing import Sequence

from narwhals.dependencies import get_pandas
from narwhals.dependencies import get_polars


def remove_prefix(text: str, prefix: str) -> str:
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text  # pragma: no cover


def remove_suffix(text: str, suffix: str) -> str:  # pragma: no cover
    if text.endswith(suffix):
        return text[: -len(suffix)]
    return text  # pragma: no cover


def flatten(args: Any) -> list[Any]:
    if not args:
        return []
    if len(args) == 1 and _is_iterable(args[0]):
        return args[0]  # type: ignore[no-any-return]
    return args  # type: ignore[no-any-return]


def _is_iterable(arg: Any | Iterable[Any]) -> bool:
    from narwhals.series import Series

    if (pd := get_pandas()) is not None and isinstance(arg, (pd.Series, pd.DataFrame)):
        msg = f"Expected Narwhals class or scalar, got: {type(arg)}. Perhaps you forgot a `nw.from_native` somewhere?"
        raise TypeError(msg)
    if (pl := get_polars()) is not None and isinstance(
        arg, (pl.Series, pl.Expr, pl.DataFrame, pl.LazyFrame)
    ):
        msg = f"Expected Narwhals class or scalar, got: {type(arg)}. Perhaps you forgot a `nw.from_native` somewhere?"
        raise TypeError(msg)

    return isinstance(arg, Iterable) and not isinstance(arg, (str, bytes, Series))


def parse_version(version: Sequence[str | int]) -> tuple[int, ...]:
    """Simple version parser; split into a tuple of ints for comparison."""
    # lifted from Polars
    if isinstance(version, str):  # pragma: no cover
        version = version.split(".")
    return tuple(int(re.sub(r"\D", "", str(v))) for v in version)


def isinstance_or_issubclass(obj: Any, cls: Any) -> bool:
    return isinstance(obj, cls) or issubclass(obj, cls)


def validate_same_library(items: Iterable[Any]) -> None:
    if all(item._is_polars for item in items):
        return
    if all(hasattr(item._dataframe, "_implementation") for item in items) and (
        len({item._dataframe._implementation for item in items}) == 1
    ):
        return
    raise NotImplementedError("Cross-library comparisons aren't supported")
