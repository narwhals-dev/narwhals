from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw
import narwhals._plan as nwp
from narwhals._plan import selectors as ncs
from narwhals._utils import Version
from tests.plan.utils import assert_equal_data, dataframe

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tests.conftest import Data

pytest.importorskip("pyarrow")
pytest.importorskip("numpy")
import numpy as np


@pytest.fixture
def data() -> Data:
    return {
        "a": ["A", "B", "A"],
        "b": [1, 2, 3],
        "c": [9, 2, 4],
        "d": [8, 7, 8],
        "e": [None, 9, 7],
        "f": [True, False, None],
        "g": [False, None, False],
        "h": [None, None, True],
        "i": [None, None, None],
        "j": [12.1, None, 4.0],
        "k": [42, 10, None],
        "l": [4, 5, 6],
        "m": [0, 1, 2],
        "n": ["dogs", "cats", None],
        "o": ["play", "swim", "walk"],
    }


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        pytest.param(
            [
                nwp.col("a")
                .alias("...")
                .map_batches(
                    lambda s: s.from_iterable(
                        [*((len(s) - 1) * [type(s.dtype).__name__.lower()]), "last"],
                        version=Version.MAIN,
                        name="funky",
                    ),
                    is_elementwise=True,
                ),
                nwp.col("a"),
            ],
            {"funky": ["string", "string", "last"], "a": ["A", "B", "A"]},
            id="series",
        ),
        pytest.param(
            nwp.col("b")
            .map_batches(lambda s: s.to_numpy() + 1, nw.Float64(), is_elementwise=True)
            .sum(),
            {"b": [9.0]},
            id="numpy",
        ),
        pytest.param(
            ncs.by_name("b", "c", "d")
            .map_batches(lambda s: np.append(s.to_numpy(), [10, 2]), is_elementwise=True)
            .sort(),
            {"b": [1, 2, 2, 3, 10], "c": [2, 2, 4, 9, 10], "d": [2, 7, 8, 8, 10]},
            id="selector",
        ),
        pytest.param(
            nwp.col("j", "k")
            .fill_null(15)
            .map_batches(lambda s: (s.to_numpy().max()), returns_scalar=True),
            {"j": [15], "k": [42]},
            id="returns_scalar",
        ),
    ],
)
def test_map_batches(
    data: Data, expr: nwp.Expr | Sequence[nwp.Expr], expected: Data
) -> None:
    result = dataframe(data).select(expr)
    assert_equal_data(result, expected)
