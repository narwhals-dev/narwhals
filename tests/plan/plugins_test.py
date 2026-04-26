from __future__ import annotations

import pytest

from narwhals._plan.compliant.plugins import load_plugin
from tests.plan.utils import re_compile


def test_load_plugin() -> None:
    # NOTE: This operation loads the package, but does not trigger any eager imports
    # It should always be safe to do this, regardless of whether or not the backend is installed
    polars_plugin = load_plugin("polars")
    pyarrow_plugin = load_plugin("pyarrow")
    assert polars_plugin.plugin_name == "polars"
    assert pyarrow_plugin.plugin_name == "pyarrow"

    assert hasattr(polars_plugin, "implementation")
    assert polars_plugin.implementation.is_polars()  # pyright: ignore[reportAttributeAccessIssue]

    assert hasattr(pyarrow_plugin, "implementation")
    assert pyarrow_plugin.implementation.is_pyarrow()  # pyright: ignore[reportAttributeAccessIssue]

    with pytest.raises(NotImplementedError, match=r"not yet"):
        load_plugin("modin")

    with pytest.raises(
        TypeError, match=re_compile(r"Unsupported `backend` .+got: 'i dont exist'")
    ):
        load_plugin("i dont exist")


XFAIL_TODO = pytest.mark.xfail(reason="TODO", raises=NotImplementedError)


# TODO @dangotbanned: Cover when connected up to the rest
@XFAIL_TODO
def test_plugin_is_loaded() -> None:  # pragma: no cover
    raise NotImplementedError


# TODO @dangotbanned: Cover when connected up to the rest
@XFAIL_TODO
def test_plugin_is_available() -> None:  # pragma: no cover
    raise NotImplementedError
