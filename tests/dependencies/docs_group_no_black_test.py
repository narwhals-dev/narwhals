from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    not Path(__file__).resolve().parents[2].joinpath("pyproject.toml").exists(),
    reason="pyproject.toml not found at repo root",
)


def _read_docs_group() -> list[str]:
    import tomllib

    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    with pyproject.open("rb") as f:
        data = tomllib.load(f)
    deps = data.get("dependency-groups", {})
    docs = deps.get("docs") or []
    return [entry for entry in docs if isinstance(entry, str)]


def test_docs_group_does_not_pull_black() -> None:
    """Regression test for narwhals#3676.

    The `docs` dependency group must not pin `black` as a direct
    dependency. mkdocstrings prefers `black` for code-block formatting
    when installed, which conflicts with project style and adds an
    avoidable heavy dependency. Since mkdocstrings 1.13.0 (2024)
    supports `ruff` as a formatter, this project uses ruff and does
    not need `black` in its docs install set.
    """
    docs_deps = _read_docs_group()
    offenders = [
        d
        for d in docs_deps
        if d.split(">=", 1)[0]
        .split("==", 1)[0]
        .split("<", 1)[0]
        .split(">", 1)[0]
        .strip()
        .lower()
        == "black"
    ]
    assert not offenders, (
        "`black` should not be a direct dependency in the [docs] group. "
        f"Found: {offenders}. See https://github.com/narwhals-dev/narwhals/issues/3676"
    )
