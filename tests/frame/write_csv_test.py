from __future__ import annotations

from typing import TYPE_CHECKING

import narwhals as nw
from tests.utils import ConstructorEager, is_windows

if TYPE_CHECKING:
    import pytest


def test_write_csv(
    nw_eager_constructor: ConstructorEager, tmpdir: pytest.TempdirFactory
) -> None:
    data = {"a": [1, 2, 3]}
    path = tmpdir / "foo.csv"  # type: ignore[operator]
    result = nw.from_native(nw_eager_constructor(data), eager_only=True).write_csv(
        str(path)
    )
    assert path.exists()
    assert result is None
    result = nw.from_native(nw_eager_constructor(data), eager_only=True).write_csv()
    if is_windows():  # pragma: no cover
        result = result.replace("\r\n", "\n")
    if "pyarrow_table" in str(nw_eager_constructor):
        assert result == '"a"\n1\n2\n3\n'
    else:
        assert result == "a\n1\n2\n3\n"
