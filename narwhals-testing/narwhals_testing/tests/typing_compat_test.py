"""Ensuring backports/extensions to new `typing` features are understood correctly."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Literal

import pytest

from narwhals._typing_compat import assert_never


def test_assert_never() -> None:
    pattern = re.compile(
        r"expected.+unreachable.+got.+'a'.+report.+issue.+github.+narwhals",
        re.DOTALL | re.IGNORECASE,
    )
    some: Literal["a"] = "a"
    if some != "a":
        assigned = "b"
        assert_never(assigned)
    else:
        assigned = some
    if not TYPE_CHECKING:
        # NOTE: Trying to avoid the assert influencing narrowing
        assert assigned == "a"
    with pytest.raises(AssertionError, match=pattern):
        assert_never(assigned)  # type: ignore[arg-type]
