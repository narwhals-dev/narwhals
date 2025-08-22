from __future__ import annotations

import sys

import pytest

import narwhals as nw


@pytest.mark.skipif(sys.version_info < (3, 10), reason="3.10+ required for entrypoints")
def test_plugin() -> None:
    pytest.importorskip("test_plugin")
    df_native = {"a": [1, 1, 2], "b": [4, 5, 6]}
    lf = nw.from_native(df_native)  # pyright: ignore[reportCallIssue, reportArgumentType]
    assert lf.columns == ["a", "b"]
