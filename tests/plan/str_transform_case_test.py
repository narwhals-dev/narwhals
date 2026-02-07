from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals._plan as nwp
from tests.plan.utils import assert_equal_data, dataframe

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

    StrData: TypeAlias = dict[str, list[str]]


@pytest.fixture(scope="module")
def data() -> StrData:
    return {
        "a": [
            "e.t. phone home",
            "they're bill's friends from the UK",
            "to infinity,and BEYOND!",
            "with123numbers",
            "__dunder__score_a1_.2b ?three",
        ]
    }


@pytest.fixture(scope="module")
def data_lower(data: StrData) -> StrData:
    return {"a": [*data["a"], "SPECIAL CASE ß", "ΣPECIAL CAΣE"]}


@pytest.fixture(scope="module")
def expected_title(data: StrData) -> StrData:
    return {"a": [s.title() for s in data["a"]]}


@pytest.fixture(scope="module")
def expected_upper(data: StrData) -> StrData:
    return {"a": [s.upper() for s in data["a"]]}


@pytest.fixture(scope="module")
def expected_lower(data_lower: StrData) -> StrData:
    return {"a": [s.lower() for s in data_lower["a"]]}


def test_str_to_titlecase(data: StrData, expected_title: StrData) -> None:
    result = dataframe(data).select(nwp.col("a").str.to_titlecase())
    assert_equal_data(result, expected_title)


def test_str_to_uppercase(data: StrData, expected_upper: StrData) -> None:
    result = dataframe(data).select(nwp.col("a").str.to_uppercase())
    assert_equal_data(result, expected_upper)


def test_str_to_lowercase(data_lower: StrData, expected_lower: StrData) -> None:
    result = dataframe(data_lower).select(nwp.col("a").str.to_lowercase())
    assert_equal_data(result, expected_lower)
