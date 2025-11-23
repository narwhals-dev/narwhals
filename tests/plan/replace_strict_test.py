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


XFAIL_DEFAULT = pytest.mark.xfail(
    reason="Not Implemented `replace_strict(default=...)` yet", raises=ValueError
)


# TODO @dangotbanned: Share more of the case generation logic from `basic_cases`
@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        # test_replace_strict_expr_with_default
        pytest.param(
            nwp.col("int").replace_strict(
                [1, 2], ["one", "two"], default=nwp.lit("other"), return_dtype=nw.String
            ),
            {"int": ["one", "two", "other", "other"]},
            marks=XFAIL_DEFAULT,
            id="non-null-1",
        ),
        pytest.param(
            nwp.col("int").replace_strict([1, 2], ["one", "two"], default="other"),
            {"int": ["one", "two", "other", "other"]},
            marks=XFAIL_DEFAULT,
            id="non-null-2",
        ),
        # test_replace_strict_with_default_and_nulls
        pytest.param(
            nwp.col("int-null").replace_strict(
                [1, 2], [10, 20], default=99, return_dtype=nw.Int64
            ),
            {"int-null": [10, 20, 99, 99]},
            marks=XFAIL_DEFAULT,
            id="null-1",
        ),
        pytest.param(
            nwp.col("int-null").replace_strict([1, 2], [10, 20], default=99),
            {"int-null": [10, 20, 99, 99]},
            marks=XFAIL_DEFAULT,
            id="null-2",
        ),
        # test_replace_strict_with_default_mapping
        pytest.param(
            nwp.col("int").replace_strict(
                {1: "one", 2: "two", 3: None}, default="other", return_dtype=nw.String()
            ),
            {"int": ["one", "two", None, "other"]},
            marks=XFAIL_DEFAULT,
            # shouldn't be an independent case, the mapping isn't the default
            id="replace_strict_with_default_mapping",
        ),
        # test_replace_strict_with_expressified_default
        pytest.param(
            nwp.col("int").replace_strict(
                {1: "one", 2: "two"}, default=nwp.col("str-alt"), return_dtype=nw.String
            ),
            {"int": ["one", "two", "orca", "vaquita"]},
            marks=XFAIL_DEFAULT,
            id="column",
        ),
        # test_mapping_key_not_in_expr
        pytest.param(
            nwp.col("int").replace_strict(
                {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}, default="hundred"
            ),
            {"int": ["one", "two", "three", "four"]},
            id="mapping_key_not_in_expr",
        ),
    ],
)
def test_replace_strict_expr_default(data: Data, expr: nwp.Expr, expected: Data) -> None:
    result = dataframe(data).select(expr)
    assert_equal_data(result, expected)
