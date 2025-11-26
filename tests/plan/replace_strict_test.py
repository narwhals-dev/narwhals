from __future__ import annotations

from itertools import chain
from typing import TYPE_CHECKING, Any, Literal

import pytest

import narwhals as nw
import narwhals._plan as nwp
from narwhals._utils import no_default
from narwhals.exceptions import InvalidOperationError
from tests.plan.utils import assert_equal_data, dataframe

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence

    from _pytest.mark import ParameterSet
    from typing_extensions import TypeAlias

    from narwhals._plan.typing import IntoExpr
    from narwhals._typing import NoDefault
    from narwhals.typing import IntoDType, NonNestedLiteral
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


def cases(
    column: Literal["str", "int", "str-null", "int-null", "str-alt"],
    replacements: Mapping[Any, Any],
    return_dtypes: Iterable[IntoDType | None],
    *,
    default: IntoExpr | NoDefault = no_default,
    expected: list[NonNestedLiteral] | None = None,
    marks: pytest.MarkDecorator | Collection[pytest.MarkDecorator | pytest.Mark] = (),
) -> Iterator[ParameterSet]:
    old, new = list(replacements), tuple(replacements.values())
    base = nwp.col(column)
    alt_name = f"{column}_seqs"
    alt = nwp.col(column).alias(alt_name)
    if expected:
        expected_m = {column: expected, alt_name: expected}
    else:
        expected_m = {column: list(new), alt_name: list(new)}
    if default is no_default:
        suffix = ""
    else:
        tp = type(default._ir) if isinstance(default, nwp.Expr) else type(default)
        suffix = f"-default-{tp.__name__}"

    for dtype in return_dtypes:
        exprs = (
            base.replace_strict(replacements, default=default, return_dtype=dtype),
            alt.replace_strict(old, new, default=default, return_dtype=dtype),
        )
        schema = {column: dtype, alt_name: dtype} if dtype else None
        id = f"{column}-{dtype}{suffix}"
        yield pytest.param(exprs, expected_m, schema, id=id, marks=marks)


@pytest.mark.parametrize(
    ("exprs", "expected", "schema"),
    chain(
        cases(
            "str",
            {"one": 1, "two": 2, "three": 3, "four": 4},
            [nw.Int8, nw.Float32, None],
        ),
        cases("int", {1: "one", 2: "two", 3: "three", 4: "four"}, [nw.String(), None]),
        cases(
            "int",
            {1: "one", 2: "two"},
            [nw.String, None],
            default=nwp.lit("other"),
            expected=["one", "two", "other", "other"],
        ),
        cases(
            "int-null",
            {1: 10, 2: 20},
            [nw.Int64, None],
            default=99,
            expected=[10, 20, 99, 99],
        ),
        cases(
            "int",
            {1: "one", 2: "two", 3: None},
            [nw.String, None],
            default="other",
            expected=["one", "two", None, "other"],
        ),
        cases(
            "int",
            {1: "one", 2: "two"},
            [nw.String, None],
            default=nwp.col("str-alt"),
            expected=["one", "two", "orca", "vaquita"],
        ),
        cases(
            "int",
            {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"},
            [None],
            default="hundred",
            expected=["one", "two", "three", "four"],
        ),
    ),
)
def test_replace_strict_expr(
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


def test_replace_strict_scalar(data: Data) -> None:
    df = dataframe(data)
    expr = (
        nwp.col("str-null")
        .first()
        .replace_strict({"one": 1, "two": 2, "three": 3, "four": 4})
    )
    assert_equal_data(df.select(expr), {"str-null": [1]})

    int_null = nwp.col("int-null")
    repl_ints = {1: 10, 2: 20, 4: 40}

    expr = int_null.last().replace_strict(repl_ints, default=999)
    assert_equal_data(df.select(expr), {"int-null": [40]})

    expr = int_null.sort(nulls_last=True).last().replace_strict(repl_ints, default=999)
    assert_equal_data(df.select(expr), {"int-null": [999]})
