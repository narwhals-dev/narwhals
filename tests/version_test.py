from __future__ import annotations

import re
from collections.abc import Callable
from importlib.metadata import distribution
from os import environ
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw
from narwhals._utils import parse_version


def is_in_ci() -> bool:
    # https://docs.github.com/en/actions/reference/workflows-and-actions/variables#default-environment-variables
    return environ.get("CI", "") == "true" and environ.get("GITHUB_ACTIONS", "") == "true"


def test_package_version() -> None:
    content = Path("pyproject.toml").read_text("utf-8")
    match = re.search(r'version = "(.*)"', content)
    assert match is not None
    pyproject_version = match.group(1)

    version = nw.__version__
    assert isinstance(version, str)
    version_comp = parse_version(version)

    msg = (
        f"Expected `nw.__version__` to match `pyproject.toml`, unless testing locally.\n\n"
        f"nw.__version__={version!r}\npyproject.toml={pyproject_version!r}"
    )
    if is_in_ci():
        assert version == pyproject_version, msg
    else:  # pragma: no cover
        # NOTE: metadata from venv may be outdated (https://github.com/narwhals-dev/narwhals/pull/3130#issuecomment-3291578373)
        assert version_comp <= parse_version(pyproject_version)
    dist_version = distribution("narwhals").version
    assert version == dist_version


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
