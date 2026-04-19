from __future__ import annotations

from typing import TYPE_CHECKING

import narwhals as nw
from narwhals.stable.v1.dependencies import is_narwhals_lazyframe
from tests.utils import Constructor

if TYPE_CHECKING:
    from tests.utils import Constructor


def test_is_narwhals_lazyframe(constructor: Constructor) -> None:
    lf = constructor({"a": [1, 2, 3]})

    assert is_narwhals_lazyframe(nw.from_native(lf).lazy())
    assert not is_narwhals_lazyframe(lf)
