from __future__ import annotations

from enum import auto

import pytest

from narwhals._enum import NoAutoEnum


def test_no_auto_enum() -> None:
    with pytest.raises(
        ValueError,
        match=r"Creating values with `auto\(\)` is not allowed. Please provide a value manually instead.",
    ):

        class TestWithAuto(NoAutoEnum):
            AUTO = auto()
