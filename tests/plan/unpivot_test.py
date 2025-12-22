from __future__ import annotations

from typing import TYPE_CHECKING, Final

import pytest

import narwhals as nw
from narwhals import _plan as nwp
from narwhals._plan import selectors as ncs
from tests.plan.utils import assert_equal_data, dataframe
from tests.utils import PYARROW_VERSION

if TYPE_CHECKING:
    from narwhals._plan.typing import ColumnNameOrSelector, OneOrIterable
    from tests.conftest import Data


@pytest.fixture(scope="module")
def data() -> Data:
    return {"a": [7, 8, 9], "b": [1, 3, 5], "c": [2, 4, 6]}


A: Final = [7, 8, 9]
B: Final = [1, 3, 5]
C: Final = [2, 4, 6]

VAR = "variable"
VALUE = "value"

a = ncs.first()
b = ncs.by_name("b")
c = ncs.last()


@pytest.mark.parametrize(
    ("on", "index", "expected"),
    [
        ("b", [a], {"a": A, VAR: ["b", "b", "b"], VALUE: B}),
        (
            ["b", c],
            a,
            {"a": [*A, *A], VAR: ["b", "b", "b", "c", "c", "c"], VALUE: [*B, *C]},
        ),
        (
            None,
            ["a"],
            {"a": [*A, *A], VAR: ["b", "b", "b", "c", "c", "c"], VALUE: [*B, *C]},
        ),
        ([b | c], None, {VAR: ["b", "b", "b", "c", "c", "c"], VALUE: [*B, *C]}),
        (
            None,
            None,
            {VAR: ["a", "a", "a", "b", "b", "b", "c", "c", "c"], VALUE: [*A, *B, *C]},
        ),
    ],
)
def test_unpivot(
    data: Data,
    on: OneOrIterable[ColumnNameOrSelector] | None,
    index: OneOrIterable[ColumnNameOrSelector] | None,
    expected: Data,
) -> None:
    sort_columns = [VAR] if index is None else [VAR, "a"]
    result = dataframe(data).unpivot(on, index=index).sort(sort_columns)
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("variable_name", "value_name"),
    [
        ("", "custom_value_name"),
        ("custom_variable_name", ""),
        ("custom_variable_name", "custom_value_name"),
    ],
)
def test_unpivot_var_value_names(data: Data, variable_name: str, value_name: str) -> None:
    result = dataframe(data).unpivot(
        ~ncs.first(), index=["a"], variable_name=variable_name, value_name=value_name
    )
    assert result.collect_schema().names()[-2:] == [variable_name, value_name]


def test_unpivot_default_var_value_names(data: Data) -> None:
    result = dataframe(data).unpivot(nwp.nth(1, 2).meta.as_selector(), index=ncs.first())
    assert result.collect_schema().names()[-2:] == [VAR, VALUE]


@pytest.mark.xfail(PYARROW_VERSION < (14, 0, 0), reason="pyarrow<14")
def test_unpivot_mixed_types() -> None:
    df = dataframe({"idx": [0, 1], "a": [1, 2], "b": [1.5, 2.5]})
    result = df.unpivot(["a", "b"], index="idx")
    assert result.collect_schema().dtypes() == [nw.Int64(), nw.String(), nw.Float64()]
