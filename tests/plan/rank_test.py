from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals._plan as nwp
from tests.plan.utils import assert_equal_data, dataframe

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from typing_extensions import TypeAlias

    from narwhals.typing import RankMethod

Data: TypeAlias = "dict[str, Sequence[float | None]]"

ASC = False
DESC = True


@pytest.fixture(params=["average", "min", "max", "dense", "ordinal"])
def rank_method(request: pytest.FixtureRequest) -> RankMethod:
    method: RankMethod = request.param
    request.applymarker(
        pytest.mark.xfail(
            method == "average",
            reason="rank('average') not yet implemented for pyarrow",
            raises=NotImplementedError,
        )
    )
    return method


def _generate_data() -> Iterator[Data]:
    a_int = [3, 6, 1, 1, None, 6]
    a_float = [3.1, 6.1, 1.5, 1.5, None, 6.1]
    for column in (a_int, a_float):
        yield {"a": column, "b": [1, 1, 2, 1, 2, 2], "i": [1, 2, 3, 4, 5, 6]}


@pytest.fixture(params=_generate_data(), scope="module", ids=["int", "float"])
def data(request: pytest.FixtureRequest) -> Data:
    data_: Data = request.param
    return data_


EXPECTED: Mapping[tuple[RankMethod, bool], Sequence[float | None]] = {
    ("average", ASC): [3.0, 4.5, 1.5, 1.5, None, 4.5],
    ("average", DESC): [3.0, 1.5, 4.5, 4.5, None, 1.5],
    ("min", ASC): [3, 4, 1, 1, None, 4],
    ("min", DESC): [3, 1, 4, 4, None, 1],
    ("max", ASC): [3, 5, 2, 2, None, 5],
    ("max", DESC): [3, 2, 5, 5, None, 2],
    ("dense", ASC): [2, 3, 1, 1, None, 3],
    ("dense", DESC): [2, 1, 3, 3, None, 1],
    ("ordinal", ASC): [3, 4, 1, 2, None, 5],
    ("ordinal", DESC): [3, 1, 4, 5, None, 2],
}


@pytest.mark.parametrize("descending", [ASC, DESC], ids=["asc", "desc"])
def test_rank_expr(rank_method: RankMethod, data: Data, *, descending: bool) -> None:
    result = dataframe(data).select(nwp.col("a").rank(rank_method, descending=descending))
    assert_equal_data(result, {"a": EXPECTED[rank_method, descending]})
