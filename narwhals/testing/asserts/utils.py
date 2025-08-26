from __future__ import annotations

from typing import Any, NoReturn

from narwhals.dependencies import is_narwhals_series


def raise_assertion_error(
    objects: str, detail: str, left: Any, right: Any, *, cause: Exception | None = None
) -> NoReturn:
    """Raise a detailed assertion error."""
    __tracebackhide__ = True

    trailing_left = "\n" if is_narwhals_series(left) else " "
    trailing_right = "\n" if is_narwhals_series(right) else " "

    msg = (
        f"{objects} are different ({detail})\n"
        f"[left]:{trailing_left}{left}\n"
        f"[right]:{trailing_right}{right}"
    )
    raise AssertionError(msg) from cause
