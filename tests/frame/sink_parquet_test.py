from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw
from tests.utils import PANDAS_VERSION

if TYPE_CHECKING:
    from tests.utils import Constructor

pytest.importorskip("pyarrow")

data = {"a": [1, 2, 3]}


@pytest.mark.skipif(PANDAS_VERSION < (2, 0, 0), reason="too old for pyarrow")
@pytest.mark.filterwarnings("ignore:.*is_sparse is deprecated:DeprecationWarning")
def test_sink_parquet(constructor: Constructor, tmpdir: pytest.TempdirFactory) -> None:
    path = tmpdir / "foo.parquet"  # type: ignore[operator]
    df = nw.from_native(constructor(data))
    df.lazy().sink_parquet(str(path))
    assert path.exists()
