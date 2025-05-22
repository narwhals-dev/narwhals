from __future__ import annotations

import warnings
from typing import Any

from narwhals.functions import _get_deps_info, _get_sys_info, show_versions


def test_get_sys_info() -> None:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        show_versions()
        sys_info = _get_sys_info()

    assert "python" in sys_info
    assert "executable" in sys_info
    assert "machine" in sys_info


def test_get_deps_info() -> None:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        show_versions()
        deps_info = _get_deps_info()

    assert "narwhals" in deps_info
    assert "pandas" in deps_info
    assert "polars" in deps_info
    assert "cudf" in deps_info
    assert "modin" in deps_info
    assert "pyarrow" in deps_info
    assert "numpy" in deps_info


def test_show_versions(capsys: Any) -> None:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        show_versions()
        out, _ = capsys.readouterr()

    assert "python" in out
    assert "machine" in out
    assert "pandas" in out
    assert "polars" in out
