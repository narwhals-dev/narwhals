"""Pytest plugin to run narwhals' own test suite against `narwhals-dict`.

Usage, from the narwhals repository root:

    uv run pytest tests -p narwhals_dict.testing --use-external-constructor

`--use-external-constructor` makes narwhals' `tests/conftest.py` skip its own
constructor parametrization; the hooks below then parametrize the `constructor`
and `constructor_eager` fixtures with a plain-dict constructor instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from importlib.metadata import EntryPoints

    from narwhals_dict.typing import DictFrame

CONSTRUCTORS_TO_SKIP = ("constructor_pandas_like",)


def dict_constructor(obj: dict[str, Any]) -> DictFrame:
    return {name: list(values) for name, values in obj.items()}


@pytest.fixture(autouse=True)
def _plugin_discovery_order(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Discover `narwhals-dict` first in constructor tests, and last in every other test.

    The in-repo `test-plugin` also claims every `dict` (returning a lazy-only
    stub), and whichever entry point loads first wins in
    `narwhals.plugins._iter_from_native`. Tests that build frames through the
    `constructor`/`constructor_eager` fixtures must get `narwhals-dict`, while the
    rest are written for `test-plugin`, the only plugin installed in CI.
    Installed entry-point order follows the filesystem, so it is pinned either way.
    """
    from importlib.metadata import entry_points

    from narwhals import plugins

    dict_first = not {"constructor", "constructor_eager"}.isdisjoint(request.fixturenames)
    eps = sorted(
        entry_points(group="narwhals.plugins"),
        key=lambda entry_point: (entry_point.name == "narwhals-dict") != dict_first,
    )

    def _discover_entrypoints() -> EntryPoints:
        return eps  # type: ignore[return-value]

    monkeypatch.setattr(plugins, "_discover_entrypoints", _discover_entrypoints)


@pytest.hookimpl(tryfirst=True)
def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "constructor_eager" in metafunc.fixturenames:
        metafunc.parametrize("constructor_eager", [dict_constructor], ids=["dict"])
    elif "constructor" in metafunc.fixturenames:
        metafunc.parametrize("constructor", [dict_constructor], ids=["dict"])
    for fixture in CONSTRUCTORS_TO_SKIP:
        if fixture in metafunc.fixturenames:
            metafunc.parametrize(fixture, [], ids=[])
