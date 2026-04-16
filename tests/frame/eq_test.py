from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw

if TYPE_CHECKING:
    from tests.utils import Constructor


def test_eq_neq_raise(constructor: Constructor) -> None:
    with pytest.raises(NotImplementedError, match="please use expressions"):
        nw.from_native(constructor({"a": [1, 2, 3]})) == 0  # noqa: B015
    with pytest.raises(NotImplementedError, match="please use expressions"):
        nw.from_native(constructor({"a": [1, 2, 3]})) != 0  # noqa: B015
