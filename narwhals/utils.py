from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Iterable

if TYPE_CHECKING:
    from narwhals.spec import IntoExpr


def remove_prefix(text: str, prefix: str) -> str:
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def flatten_into_expr(*args: IntoExpr | Iterable[IntoExpr]) -> list[IntoExpr]:
    out: list[IntoExpr] = []
    for arg in args:
        if isinstance(arg, (list, tuple)):
            out.extend(arg)
        else:
            out.append(arg)  # type: ignore[arg-type]
    return out


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
