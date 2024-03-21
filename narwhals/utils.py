from __future__ import annotations

import re
from typing import Any
from typing import Iterable
from typing import Sequence


def remove_prefix(text: str, prefix: str) -> str:
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def flatten(args: Any) -> list[Any]:
    if not args:
        return []
    if len(args) == 1 and _is_iterable(args[0]):
        return args[0]  # type: ignore[no-any-return]
    return args  # type: ignore[no-any-return]


def _is_iterable(arg: Any | Iterable[Any]) -> bool:
    from narwhals.series import Series

    return isinstance(arg, Iterable) and not isinstance(arg, (str, bytes, Series))


def flatten_str(*args: str | Iterable[str]) -> list[str]:
    out: list[str] = []
    for arg in args:
        if isinstance(arg, str):
            out.append(arg)
        else:
            for item in arg:
                if not isinstance(item, str):
                    msg = f"Expected str, got {type(item)}"
                    raise TypeError(msg)
                out.append(item)
    return out


def flatten_bool(*args: bool | Iterable[bool]) -> list[bool]:
    out: list[bool] = []
    for arg in args:
        if isinstance(arg, bool):
            out.append(arg)
        else:
            for item in arg:
                if not isinstance(item, bool):
                    msg = f"Expected str, got {type(item)}"
                    raise TypeError(msg)
                out.append(item)
    return out


def parse_version(version: Sequence[str | int]) -> tuple[int, ...]:
    """Simple version parser; split into a tuple of ints for comparison."""
    if isinstance(version, str):
        version = version.split(".")
    return tuple(int(re.sub(r"\D", "", str(v))) for v in version)
