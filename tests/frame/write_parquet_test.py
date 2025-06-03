from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw
from tests.utils import PANDAS_VERSION

if TYPE_CHECKING:
    from tests.utils import ConstructorEager

data = {"a": [1, 2, 3]}


@pytest.mark.skipif(PANDAS_VERSION < (2, 0, 0), reason="too old for pyarrow")
@pytest.mark.filterwarnings("ignore:.*is_sparse is deprecated:DeprecationWarning")
def test_write_parquet(
    constructor_eager: ConstructorEager, tmpdir: pytest.TempdirFactory
) -> None:
    path = tmpdir / "foo.parquet"  # type: ignore[operator]
    nw.from_native(constructor_eager(data), eager_only=True).write_parquet(str(path))
    assert path.exists()
