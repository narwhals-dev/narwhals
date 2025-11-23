from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw
import narwhals._plan as nwp
from narwhals.exceptions import InvalidOperationError
from tests.plan.utils import assert_equal_data, dataframe

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence

    from _pytest.mark import ParameterSet
    from typing_extensions import TypeAlias

    from narwhals._typing import NoDefault
    from narwhals.typing import IntoDType
    from tests.conftest import Data

pytest.importorskip("pyarrow")


Old: TypeAlias = "Sequence[Any] | Mapping[Any, Any]"
New: TypeAlias = "Sequence[Any] | NoDefault"


@pytest.fixture(scope="module")
def data() -> Data:
    return {
        "str": ["one", "two", "three", "four"],
        "int": [1, 2, 3, 4],
        "str-null": ["one", None, "three", "four"],
        "int-null": [1, 2, None, 4],
        "str-alt": ["beluga", "narwhal", "orca", "vaquita"],
    }


def basic_cases(
    column: str,
    replacements: Mapping[Any, Any],
    return_dtypes: Iterable[IntoDType | None],
) -> Iterator[ParameterSet]:
    old, new = list(replacements), tuple(replacements.values())
    values = list(new)
    base = nwp.col(column)
    alt_name = f"{column}_seqs"
    alt = nwp.col(column).alias(alt_name)
    expected = {column: values, alt_name: values}
    for dtype in return_dtypes:
        exprs = (
            base.replace_strict(replacements, return_dtype=dtype),
            alt.replace_strict(old, new, return_dtype=dtype),
        )
        schema = {column: dtype, alt_name: dtype} if dtype is not None else None
        yield pytest.param(exprs, expected, schema, id=f"{column}-{dtype}")


@pytest.mark.parametrize(
    ("exprs", "expected", "schema"),
    [
        *basic_cases(
            "str",
            {"one": 1, "two": 2, "three": 3, "four": 4},
            [nw.Int8, nw.Float32, None],
        ),
        *basic_cases(
            "int", {1: "one", 2: "two", 3: "three", 4: "four"}, [nw.String(), None]
        ),
    ],
)
def test_replace_strict_expr_basic(
    data: Data,
    exprs: Iterable[nwp.Expr],
    expected: Data,
    schema: Mapping[str, IntoDType] | None,
) -> None:
    result = dataframe(data).select(exprs)
    assert_equal_data(result, expected)
    if schema is not None:
        assert result.collect_schema() == schema


@pytest.mark.parametrize(
    "expr",
    [
        nwp.col("int").replace_strict([1, 3], [3, 4]),
        nwp.col("str-null").replace_strict({"one": "two", "four": "five"}),
    ],
)
def test_replace_strict_expr_non_full(data: Data, expr: nwp.Expr) -> None:
    with pytest.raises(
        (ValueError, InvalidOperationError), match=r"did not replace all non-null"
    ):
        dataframe(data).select(expr)
