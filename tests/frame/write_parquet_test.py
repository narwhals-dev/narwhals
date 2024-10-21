from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals.stable.v1 as nw

if TYPE_CHECKING:
    from tests.utils import ConstructorEager

data = {"a": [1, 2, 3]}


def test_write_parquet(
    constructor_eager: ConstructorEager,
    tmpdir: pytest.TempdirFactory,
    request: pytest.FixtureRequest,
    pandas_version: tuple[int, ...],
) -> None:
    if pandas_version < (2, 0, 0):
        request.applymarker(pytest.mark.skip(reason="too old for pyarrow"))
    path = tmpdir / "foo.parquet"  # type: ignore[operator]
    nw.from_native(constructor_eager(data), eager_only=True).write_parquet(str(path))
    assert path.exists()
