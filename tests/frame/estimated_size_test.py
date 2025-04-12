from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw

if TYPE_CHECKING:
    from tests.utils import ConstructorEager

data = {"a": list(range(100))}


def test_estimated_size(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)

    assert df.estimated_size("b") > 0
    assert df.estimated_size("kb") == (df.estimated_size("b") / 1024)
    assert df.estimated_size("mb") == (df.estimated_size("kb") / 1024)
    assert df.estimated_size("gb") == (df.estimated_size("mb") / 1024)
    assert df.estimated_size("tb") == (df.estimated_size("gb") / 1024)

    with pytest.raises(
        ValueError,
        match="`unit` must be one of {'b', 'kb', 'mb', 'gb', 'tb'}, got 'pizza'",
    ):
        df.estimated_size("pizza")  # type: ignore[arg-type]
