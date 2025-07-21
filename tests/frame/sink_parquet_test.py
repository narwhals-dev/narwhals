from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw

if TYPE_CHECKING:
    from tests.utils import Constructor

data = {"a": [1, 2, 3]}


@pytest.mark.filterwarnings("ignore:.*is_sparse is deprecated:DeprecationWarning")
def test_sink_parquet(constructor: Constructor, tmpdir: pytest.TempdirFactory) -> None:
    path = tmpdir / "foo.parquet"  # type: ignore[operator]
    df = nw.from_native(constructor(data))
    if any(x in str(constructor) for x in ("pandas", "pyarrow")):
        df.write_parquet(str(path))
    else:
        df.lazy().sink_parquet(str(path))
    assert path.exists()
