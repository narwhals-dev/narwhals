from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import narwhals.stable.v1 as nw

if TYPE_CHECKING:
    import pytest


def test_write_csv(constructor_eager: Any, tmpdir: pytest.TempdirFactory) -> None:
    data = {"a": [1, 2, 3]}
    path = tmpdir / "foo.parquet"  # type: ignore[operator]
    result = nw.from_native(constructor_eager(data), eager_only=True).write_csv(str(path))
    assert path.exists()
    assert result is None
    result = nw.from_native(constructor_eager(data), eager_only=True).write_csv()
    if "pyarrow_table" in str(constructor_eager):
        assert result == '"a"\n1\n2\n3\n'
    else:
        assert result == "a\n1\n2\n3\n"
