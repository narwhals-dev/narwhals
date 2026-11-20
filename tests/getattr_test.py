from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw


def test_package_getattr() -> None:
    pytest.importorskip("typing_extensions")
    from typing_extensions import assert_type

    ok = nw.__version__
    assert_type(ok, str)
    also_ok = nw.all
    assert_type(also_ok, Callable[[], nw.Expr])

    if TYPE_CHECKING:
        bad = nw.not_real  # type: ignore[attr-defined]
        assert_type(bad, Any)

    with pytest.raises(AttributeError):
        very_bad = nw.not_real  # type: ignore[attr-defined]  # noqa: F841


def test_stable_package_getattr() -> None:
    pytest.importorskip("typing_extensions")
    from typing_extensions import assert_type

    import narwhals.stable.v1 as nw_v1
    import narwhals.stable.v2 as nw_v2

    ok_v1 = nw_v1.__version__
    assert_type(ok_v1, str)
    assert ok_v1 == nw.__version__

    ok_v2 = nw_v2.__version__
    assert_type(ok_v2, str)
    assert ok_v2 == nw.__version__
