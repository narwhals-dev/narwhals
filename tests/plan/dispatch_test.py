from __future__ import annotations

import importlib
import re
from typing import TYPE_CHECKING, Any, Final

import pytest

import narwhals as nw
from narwhals import _plan as nwp
from narwhals._plan import (
    _dispatch,
    _expr_ir,
    common,
    expressions as ir,
    selectors as ncs,
)
from narwhals._plan._dispatch import DispatcherOptions, get_dispatch_name
from narwhals._plan._flags import FunctionFlags
from narwhals._plan._function import UnaryFunction
from narwhals._plan._nodes import node
from tests.plan.utils import DataFrame, assert_equal_data, re_compile

if TYPE_CHECKING:
    from typing import TypeAlias

    from pytest import FixtureRequest

    from narwhals._plan.typing import Constructs
    from tests.conftest import Data

    DispatchRaises: TypeAlias = pytest.RaisesExc[NotImplementedError | AttributeError]

DISPATCH_MODULE: Final = _dispatch


@pytest.fixture
def data() -> dict[str, Any]:
    return {
        "a": [12.1, None, 4.0],
        "b": [42, 10, None],
        "c": [4, 5, 6],
        "d": ["play", "swim", "walk"],
    }


def test_dispatch(data: Data, dataframe: DataFrame) -> None:
    df = dataframe(data)
    implemented_full = nwp.col("a").is_null()
    forgot_to_expand = (ir.NamedIR("howdy", nwp.nth(3, 4).first()._ir),)
    aliased_after_expand: tuple[ir.NamedIR] = (ir.NamedIR("b", ir.col("a").alias("b")),)
    assert_equal_data(df.select(implemented_full), {"a": [False, True, False]})

    with pytest.raises(
        TypeError,
        match=re_compile(r"ByIndex.+not.+appear.+compliant.+expand.+expr.+first"),
    ):
        df._compliant.select(forgot_to_expand)

    bad = re.escape("col('a').alias('b')")
    with pytest.raises(TypeError, match=re_compile(rf"Alias.+not.+appear.+got.+{bad}")):
        df._compliant.select(aliased_after_expand)

    # Not a narwhals method, to make sure this doesn't allow arbitrary calls
    with pytest.raises(AttributeError):
        nwp.col("a").max().to_physical()  # type: ignore[attr-defined]


# TODO @dangotbanned: Fix depending on `Function.__expr_ir_dispatch__.bind`
def test_dispatch_not_yet(data: Data, dataframe: DataFrame) -> None:
    df = dataframe(data)
    # NOTE: This will unconditionally trigger an `AttributeError`
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            ir.functions.EwmMean.__expr_ir_dispatch__,
            "bind",
            lambda _: lambda _node, _frame, _name: None,
        )
        with pytest.raises(
            NotImplementedError, match=r"ewm_mean.+is not yet implemented for"
        ):
            df.select(nwp.col("c").ewm_mean())


@pytest.mark.parametrize("enable_hints", [True, False])
def test_missing_compliant_method(
    data: Data, dataframe: DataFrame, request: FixtureRequest, *, enable_hints: bool
) -> None:
    class MissingMethod(
        UnaryFunction, dispatch=DispatcherOptions(accessor_name="str")
    ): ...

    dataframe.xfail(
        request,
        dataframe.is_polars(),
        reason="'TODO: `PolarsExpr.str(...)`'",
        raises=AssertionError if enable_hints else NotImplementedError,
    )

    df = dataframe(data)
    expr = MissingMethod().to_function_expr(ir.col("d"))
    truthy = (
        r"str\.missing_method.+has not been implemented at.+compliant-level.+"
        r"Hint.+try adding.+CompliantExpr\.str\.missing_method\(\)"
    )
    falsy = r"has no attribute.+missing_method"
    assert_dispatch_raises(df, expr, truthy, falsy, enable_hints=enable_hints)


@pytest.mark.parametrize("name", ["Expr", "Scalar"])
@pytest.mark.parametrize("enable_hints", [True, False])
def test_missing_compliant_constructor(
    data: Data, dataframe: DataFrame, name: Constructs, *, enable_hints: bool
) -> None:
    class MissingConstructor(_expr_ir.Constructor, dispatch=name):
        __slots__ = ("expr",)
        expr: ir.ExprIR = node()

    df = dataframe(data)
    expr = MissingConstructor(expr=ir.col("a"))
    truthy = (
        r"missing_constructor.+has not been implemented at.+compliant-level.+"
        rf"Hint.+try adding.+Compliant{name}\.missing_constructor\(\)"
    )
    falsy = r"has no attribute.+missing_constructor"
    assert_dispatch_raises(df, expr, truthy, falsy, enable_hints=enable_hints)


def _dispatch_raises(truthy: str, falsy: str, *, enable_hints: bool) -> DispatchRaises:
    errors = NotImplementedError if enable_hints else AttributeError
    return pytest.raises(errors, match=re_compile(truthy if enable_hints else falsy))


def assert_dispatch_raises(
    df: nwp.DataFrame, expr: ir.ExprIR, truthy: str, falsy: str, *, enable_hints: bool
) -> None:
    nw_expr = expr.to_narwhals()
    raises = _dispatch_raises(truthy, falsy, enable_hints=enable_hints)
    with pytest.MonkeyPatch.context() as mp:
        if enable_hints:
            mp.setenv(common.NW_DEV_ENV_NAME, "1")
        else:
            mp.delenv(common.NW_DEV_ENV_NAME, raising=False)
        # NOTE: The implementation works like a compile-time flag, allowing the feature to be zero-cost.
        # Updating the env var doesn't change the function, only a reload will.
        importlib.reload(DISPATCH_MODULE)
        with raises:
            df.select(nw_expr)


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (nwp.col("a"), "col"),
        (nwp.len(), "len_star"),
        (nwp.col("a").len(), "len"),
        (nwp.col("a").min().over("b"), "over"),
        (nwp.col("a").first().over(order_by="b"), "over_ordered"),
        (nwp.all_horizontal("a", "b", nwp.nth(4, 5, 6)), "all_horizontal"),
        (nwp.int_range(10), "int_range"),
        (nwp.col("a") + nwp.col("b") + 10, "binary_expr"),
        (nwp.when(nwp.col("c")).then(5).when(nwp.col("d")).then(20), "ternary_expr"),
        (nwp.col("a").rolling_sum(2), "rolling_sum"),
        (nwp.col("a").cum_sum(), "cum_sum"),
        (nwp.col("a").cat.get_categories(), "cat.get_categories"),
        (nwp.col("a").dt.timestamp(), "dt.timestamp"),
        (nwp.col("a").dt.replace_time_zone(None), "dt.replace_time_zone"),
        (nwp.col("a").list.len(), "list.len"),
        (nwp.col("a").cast(nw.String).str.starts_with("something"), ("str.starts_with")),
        (nwp.col("a").str.slice(1), ("str.slice")),
        (nwp.col("a").str.head(), ("str.slice")),
        (nwp.col("a").str.tail(), ("str.slice")),
        (nwp.col("a").struct.field("b"), "struct.field"),
        (nwp.col("a").str.zfill(4), "str.zfill"),
        (nwp.mean("a"), "mean"),
        (nwp.nth(1).first(), "first"),
        (nwp.col("a").sum(), "sum"),
        (~nwp.col("a"), "not_"),
        (nwp.col("a").drop_nulls().arg_min(), "arg_min"),
        (nwp.col("a").map_batches(lambda x: x), "map_batches"),
        pytest.param(nwp.col("a").alias("b"), "Alias", id="no_dispatch-Alias"),
        pytest.param(ncs.string(), "String", id="no_dispatch-RootSelector"),
        pytest.param(
            ncs.by_name("a") | ncs.float(),
            "BinarySelector",
            id="no_dispatch-BinarySelector",
        ),
    ],
)
def test_dispatch_name(expr: nwp.Expr, expected: str) -> None:
    assert get_dispatch_name(expr._ir) == expected


def test_sharing_expr_ir() -> None:
    expr_ir = ir.ExprIR.__expr_ir_dispatch__
    selector_ir = ir.SelectorIR.__expr_ir_dispatch__
    agg_expr = ir.AggExpr.__expr_ir_dispatch__
    root_selector = ir.RootSelector.__expr_ir_dispatch__
    first = ir.aggregation.First.__expr_ir_dispatch__

    assert issubclass(ir.SelectorIR, _expr_ir.NoDispatch)
    assert not issubclass(ir.AggExpr, _expr_ir.NoDispatch)

    # Each class has a unique instance
    assert expr_ir is not selector_ir
    assert agg_expr is not expr_ir
    assert agg_expr is not first
    assert selector_ir is not root_selector

    # Since inheritance produces different results
    assert selector_ir.name == "SelectorIR"
    assert root_selector.name == "RootSelector"
    assert first.name == "first"
    assert first.name != agg_expr.name


def test_multiple_inheritance_function() -> None:
    ACCESSOR_BIN = DispatcherOptions(accessor_name="bin")  # noqa: N806

    class ConfiguredFlagsSkip(ir.Function, dispatch="skip"):
        __function_flags__ = FunctionFlags.ELEMENTWISE

    class ConfiguredDispatch(ir.Function, dispatch=ACCESSOR_BIN): ...

    class DirectFlagsSkip(ConfiguredFlagsSkip): ...

    class DirectDispatch(ConfiguredDispatch): ...

    class MixDispatchFlagsSkip(ConfiguredDispatch, ConfiguredFlagsSkip): ...

    class MixFlagsSkipDispatch(ConfiguredFlagsSkip, ConfiguredDispatch): ...

    # Baseline for standard mro behavior
    for tp in ConfiguredFlagsSkip.__subclasses__():
        assert tp.__function_flags__ is FunctionFlags.ELEMENTWISE

    assert get_dispatch_name(ConfiguredFlagsSkip) != "configured_flags_skip"
    assert get_dispatch_name(DirectFlagsSkip) == "direct_flags_skip"

    # Most derived wins
    assert get_dispatch_name(ConfiguredDispatch) == "bin.configured_dispatch"
    assert get_dispatch_name(DirectDispatch) == "bin.direct_dispatch"
    assert get_dispatch_name(MixDispatchFlagsSkip) == "bin.mix_dispatch_flags_skip"

    # `"skip"` reflects that the definition of `ConfiguredFlagsSkip` did not specify `dispatch`
    assert get_dispatch_name(MixFlagsSkipDispatch) == "bin.mix_flags_skip_dispatch"

    # And just to be sure
    for tp in ConfiguredDispatch.__subclasses__():
        assert tp.__expr_ir_dispatch__.options.accessor_name == "bin"
