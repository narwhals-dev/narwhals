from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from packaging.version import Version as PkgVersion

import narwhals as nw
from tests.utils import PANDAS_VERSION

if TYPE_CHECKING:
    from tests.utils import Constructor

pytest.importorskip("pyarrow")

data = {"a": [1, 2, 3]}


@pytest.mark.filterwarnings("ignore:.*is_sparse is deprecated:DeprecationWarning")
def test_sink_parquet(constructor: Constructor, tmpdir: pytest.TempdirFactory) -> None:
    if "pandas" in str(constructor) and PANDAS_VERSION < PkgVersion("2"):
        pytest.skip(reason="too old for pyarrow")
    path = tmpdir / "foo.parquet"  # type: ignore[operator]
    df = nw.from_native(constructor(data))
    df.lazy().sink_parquet(str(path))
    assert path.exists()
