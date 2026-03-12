from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pytest

import narwhals as nw
from tests.conftest import TEST_EAGER_BACKENDS

if TYPE_CHECKING:
    from narwhals.typing import EagerAllowed
_HAS_IMPLEMENTATION = frozenset((nw.Implementation.PYARROW, "pyarrow"))
"""Using to filter *the source* of `eager_backend` - which includes `polars` and `pandas` when available.

For now, this lets some tests be written in a backend agnostic way.
"""


@pytest.fixture(
    scope="session", params=_HAS_IMPLEMENTATION.intersection(TEST_EAGER_BACKENDS)
)
def eager(request: pytest.FixtureRequest) -> EagerAllowed:
    result: EagerAllowed = request.param
    return result


_HAS_IMPLEMENTATION_IMPL = frozenset(
    el for el in _HAS_IMPLEMENTATION if isinstance(el, nw.Implementation)
)
"""Filtered for heavily parametric tests."""


@pytest.fixture(
    scope="session",
    params=_HAS_IMPLEMENTATION_IMPL.intersection(TEST_EAGER_BACKENDS).union([False]),
)
def eager_or_false(request: pytest.FixtureRequest) -> EagerAllowed | Literal[False]:
    result: EagerAllowed | Literal[False] = request.param
    return result
