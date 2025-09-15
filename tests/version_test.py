from __future__ import annotations

import re
from importlib.metadata import distribution
from pathlib import Path

import narwhals as nw
from narwhals._utils import parse_version


def test_package_version() -> None:
    version = nw.__version__
    dist_version = distribution("narwhals").version

    with Path("pyproject.toml").open(encoding="utf-8") as file:
        content = file.read()
        match = re.search(r'version = "(.*)"', content)
        assert match is not None
        pyproject_version = match.group(1)

    assert isinstance(version, str)
    if version != pyproject_version:
        # NOTE: metadata from venv is outdated (https://github.com/narwhals-dev/narwhals/pull/3130#issuecomment-3291578373)
        version_comp = parse_version(version)
        assert version_comp < parse_version(pyproject_version)
        assert version_comp == parse_version(dist_version)
    assert version == dist_version
