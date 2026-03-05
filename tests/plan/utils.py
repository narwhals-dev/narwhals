from __future__ import annotations

# mypy: disable-error-code="no-any-return"
import re
import threading
from collections import defaultdict
from importlib.util import find_spec
from itertools import chain
from operator import attrgetter
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    Literal,
    Protocol,
    TypedDict,
    overload,
)

import pytest

import narwhals as nw
from narwhals import _plan as nwp
from narwhals._plan import Expr, Selector, _expansion, _parse, expressions as ir
from narwhals._plan.compliant.typing import Native as NativeLazyFrame
from narwhals._plan.typing import NativeDataFrameT_co, NativeSeriesT_co
from narwhals._utils import Implementation, Version, qualified_type_name
from tests.utils import assert_equal_data as _assert_equal_data

pytest.importorskip("pyarrow")

from collections.abc import Iterable, Mapping, Sequence
from typing import TypeVar

import pyarrow as pa

if TYPE_CHECKING:
    import sys
    from collections.abc import Iterable, Mapping

    import polars as pl
    from typing_extensions import LiteralString, ReadOnly, TypeAlias

    from narwhals._plan.typing import IntoExpr, OneOrIterable, Seq
    from narwhals._typing import BackendName
    from narwhals.schema import Schema
    from narwhals.typing import (
        EagerAllowed,
        IntoBackend,
        IntoDType,
        IntoSchema,
        LazyAllowed,
    )

    if sys.version_info >= (3, 11):
        _Flags: TypeAlias = "int | re.RegexFlag"
    else:
        _Flags: TypeAlias = int

    T = TypeVar("T")
    SubList: TypeAlias = list[T] | list[T | None] | list[None] | None
    TestBackendAny: TypeAlias = "TestBackend[Any, Any, Any]"

R_co = TypeVar("R_co", covariant=True)


class _Constructor(Protocol[R_co]):
    def __call__(self, data: Any, *args: Any, **kwds: Any) -> R_co: ...


ConstructorFixtureName: TypeAlias = Literal["lazyframe", "dataframe", "series"]


class SupportProfile(TypedDict):
    """Flags declaring support for a fixture of the same name."""

    lazyframe: ReadOnly[bool]
    """Supports `lazyframe`."""
    dataframe: ReadOnly[bool]
    """Supports `dataframe`."""
    series: ReadOnly[bool]
    """Supports `series`."""


Incomplete: TypeAlias = Any
ModuleName: TypeAlias = "LiteralString"
Identifier: TypeAlias = "BackendName | ModuleName | LiteralString"
"""String used for `parametrize` test ids and backend `include`/`exclude` filters."""

UnknownBehavior: TypeAlias = Literal["raise", "ignore"]
"""How to treat `Implementation.UNKNOWN` for an operation."""


def first(*names: str) -> nwp.Expr:
    return nwp.col(*names).first()


def last(*names: str) -> nwp.Expr:
    return nwp.col(*names).last()


class Frame:
    """Schema-only `{Expr,Selector}` projection testing tool.

    Arguments:
        schema: A Narwhals Schema.

    Examples:
        >>> import narwhals as nw
        >>> import narwhals._plan.selectors as ncs
        >>> df = Frame.from_mapping(
        ...     {
        ...         "abc": nw.UInt16(),
        ...         "bbb": nw.UInt32(),
        ...         "cde": nw.Float64(),
        ...         "def": nw.Float32(),
        ...         "eee": nw.Boolean(),
        ...     }
        ... )

        Determine the columns names that expression input would select

        >>> df.project_names(ncs.numeric() - ncs.by_index(1, 2))
        ('abc', 'def')

        Assert an expression selects names in a given order

        >>> df.assert_selects(ncs.by_name("eee", "abc"), "eee", "abc")

        Raising a helpful error if something went wrong

        >>> df.assert_selects(ncs.duration(), "eee", "abc")
        Traceback (most recent call last):
        AssertionError: Projected column names do not match expected names:
        result  : ()
        expected: ('eee', 'abc')
    """

    def __init__(self, schema: nw.Schema) -> None:
        self.schema = schema
        self.columns = tuple(schema.names())

    @staticmethod
    def from_mapping(mapping: IntoSchema) -> Frame:
        """Construct from inputs accepted in `nw.Schema`."""
        return Frame(nw.Schema(mapping))

    @staticmethod
    def from_names(*column_names: str) -> Frame:
        """Construct with all `nw.Int64()`."""
        return Frame(nw.Schema((name, nw.Int64()) for name in column_names))

    @property
    def width(self) -> int:
        """Get the number of columns in the schema."""
        return len(self.columns)

    def project(
        self, exprs: OneOrIterable[IntoExpr], *more_exprs: IntoExpr
    ) -> Seq[ir.NamedIR]:
        """Parse and expand expressions into named representations.

        Arguments:
            exprs: Column(s) to select. Accepts expression input. Strings are parsed as column names,
                other non-expression inputs are parsed as literals.
            *more_exprs: Column(s) to select, specified as positional arguments.

        Note:
            `NamedIR` is the form of expression passed to the compliant-level.

        Examples:
            >>> import datetime as dt
            >>> import narwhals._plan.selectors as ncs
            >>> df = Frame.from_names("a", "b", "c", "d", "idx1", "idx2")
            >>> expr_1 = (
            ...     ncs.by_name("a", "d")
            ...     .first()
            ...     .over(ncs.by_index(range(1, 4)), order_by=ncs.matches(r"idx"))
            ... )
            >>> expr_2 = (ncs.by_name("a") | ncs.by_index(2)).abs().name.suffix("_abs")
            >>> expr_3 = dt.date(2000, 1, 1)

            >>> df.project(expr_1, expr_2, expr_3)  # doctest: +NORMALIZE_WHITESPACE
            (a=col('a').first().over(partition_by=[col('b'), col('c'), col('d')], order_by=[col('idx1'), col('idx2')]),
             d=col('d').first().over(partition_by=[col('b'), col('c'), col('d')], order_by=[col('idx1'), col('idx2')]),
             a_abs=col('a').abs(),
             c_abs=col('c').abs(),
             literal=lit(date: 2000-01-01))
        """
        expr_irs = _parse.parse_into_seq_of_expr_ir(exprs, *more_exprs)
        named_irs, _ = _expansion.prepare_projection(expr_irs, schema=self.schema)
        return named_irs

    def project_names(self, *exprs: IntoExpr) -> Seq[str]:
        named_irs = self.project(*exprs)
        return tuple(e.name for e in named_irs)

    def assert_selects(self, selector: Selector | Expr, *column_names: str) -> None:
        result = self.project_names(selector)
        expected = column_names
        assert result == expected, (
            f"Projected column names do not match expected names:\n"
            f"result  : {result!r}\n"
            f"expected: {expected!r}"
        )


def _unwrap_ir(obj: nwp.Expr | ir.ExprIR | ir.NamedIR) -> ir.ExprIR:
    if isinstance(obj, nwp.Expr):
        return obj._ir
    if isinstance(obj, ir.ExprIR):
        return obj
    if isinstance(obj, ir.NamedIR):
        return obj.expr
    raise NotImplementedError(type(obj))


def assert_expr_ir_equal(
    actual: nwp.Expr | ir.ExprIR | ir.NamedIR,
    expected: nwp.Expr | ir.ExprIR | ir.NamedIR | LiteralString,
    /,
) -> None:
    """Assert that `actual` is equivalent to `expected`.

    Arguments:
        actual: Result expression or IR to compare.
        expected: Target expression, IR, or repr to compare.

    Notes:
        Performing a repr comparison is more fragile, so should be avoided
        *unless* we raise an error at creation time.
    """
    lhs = _unwrap_ir(actual)
    if isinstance(expected, str):
        assert repr(lhs) == expected, (
            f"\nlhs:\n    {lhs!r}\n\nexpected:\n    {expected!r}"
        )
    elif isinstance(actual, ir.NamedIR) and isinstance(expected, ir.NamedIR):
        assert actual == expected, (
            f"\nactual:\n    {actual!r}\n\nexpected:\n    {expected!r}"
        )
    else:
        rhs = expected._ir if isinstance(expected, nwp.Expr) else expected
        assert lhs == rhs, f"\nlhs:\n    {lhs!r}\n\nrhs:\n    {rhs!r}"


def assert_not_selector(actual: Expr | Selector, /) -> None:
    """Assert that `actual` was converted into an `Expr`."""
    assert isinstance(actual, Expr), (
        f"Didn't expect you to pass a {qualified_type_name(actual)!r} here, got: {actual!r}"
    )
    assert not isinstance(actual, Selector), (
        f"This operation should have returned `Expr`, but got {qualified_type_name(actual)!r}\n{actual!r}"
    )


def is_expr_ir_equal(actual: Expr | ir.ExprIR, expected: Expr | ir.ExprIR, /) -> bool:
    """Return True if `actual` is equivalent to `expected`.

    Note:
        Prefer `assert_expr_ir_equal` unless you need a `bool` for branching.
    """
    return _unwrap_ir(actual) == _unwrap_ir(expected)


def named_ir(name: str, expr: nwp.Expr | ir.ExprIR, /) -> ir.NamedIR[ir.ExprIR]:
    """Helper constructor for test compare."""
    return ir.NamedIR(expr=expr._ir if isinstance(expr, nwp.Expr) else expr, name=name)


_lock = threading.Lock()


def _parse_identifiers(ids: OneOrIterable[Identifier], /) -> frozenset[Identifier]:
    return frozenset((ids,) if isinstance(ids, str) else ids)


class TestBackend(Generic[NativeLazyFrame, NativeDataFrameT_co, NativeSeriesT_co]):
    """Helper for parametrizing multiple fixtures for a single backend.

    Each backend should subclass and filling in any relevant `ClassVar`(s) & `native_*` constructors.

    `conftest.py` will take care of the rest 😄
    """

    import_or_skip_module: ClassVar[ModuleName]
    """Equivalent to[^1] the string used in `pytest.importorskip(...)`.

    [^1]: Currently passed to `importlib.util.find_spec`, as it is needed before
          test collection.
    """

    implementation: ClassVar[Implementation] = Implementation.UNKNOWN
    """Required for internal backends, plugins use the default `UNKNOWN`."""

    backend_eager: ClassVar[IntoBackend[EagerAllowed]]
    """Argument passed to `backend` or `eager` for `DataFrame`, `Series` constructors."""

    backend_lazy: ClassVar[IntoBackend[LazyAllowed]]
    """Argument passed to `backend` for `LazyFrame` constructors."""

    supports: ClassVar[SupportProfile]
    """Which fixtures the backend should populate.

    Added during `__init_subclass__`.
    """

    _BACKENDS: ClassVar[defaultdict[ModuleName, set[type[TestBackend[Any, Any]]]]] = (
        defaultdict(set)
    )

    def lazyframe(
        self, data: Mapping[str, Any], /, **kwds: Any
    ) -> nwp.LazyFrame[NativeLazyFrame]:
        return nwp.LazyFrame.from_native(self.native_lazyframe(data, **kwds))

    def dataframe(
        self, data: Mapping[str, Any], /, **kwds: Any
    ) -> nwp.DataFrame[NativeDataFrameT_co, NativeSeriesT_co]:
        return nwp.DataFrame.from_native(self.native_dataframe(data, **kwds))

    def series(
        self, values: Iterable[Any], /, **kwds: Any
    ) -> nwp.Series[NativeSeriesT_co]:
        return nwp.Series.from_native(self.native_series(values, **kwds))

    def native_lazyframe(
        self, data: Mapping[str, Any], /, **kwds: Any
    ) -> NativeLazyFrame:
        """Construct a native lazyframe."""
        msg = f"`{self.native_lazyframe.__qualname__}()` is not yet implemented"
        raise NotImplementedError(msg)

    def native_dataframe(
        self, data: Mapping[str, Any], /, **kwds: Any
    ) -> NativeDataFrameT_co:
        """Construct a native dataframe."""
        msg = f"`{self.native_dataframe.__qualname__}()` is not yet implemented"
        raise NotImplementedError(msg)

    def native_series(self, values: Iterable[Any], /, **kwds: Any) -> NativeSeriesT_co:
        """Construct a native series."""
        msg = f"`{self.native_series.__qualname__}()` is not yet implemented"
        raise NotImplementedError(msg)

    def schema_to_native(self, schema: Schema, /, **kwds: Any) -> Any:
        """Convert a narwhals schema into a native representation."""
        msg = f"`{self.schema_to_native.__qualname__}()` is not yet implemented"
        raise NotImplementedError(msg)

    def dtype_to_native(self, dtype: IntoDType, /, **kwds: Any) -> Any:
        """Convert a narwhals dtype into a native dtype."""
        msg = f"`{self.dtype_to_native.__qualname__}()` is not yet implemented"
        raise NotImplementedError(msg)

    def __repr__(self) -> str:
        return type(self).__name__

    @property
    def identifier(self) -> BackendName | ModuleName | LiteralString:
        """Base parameter for generating test ids.

        - Instance property to support variation in the same subclass
        - Should be built entirely from **literal string(s)**, to support predictable filtering
        """
        if self.implementation is Implementation.UNKNOWN:
            return self.import_or_skip_module
        return self.implementation.value

    def try_get_constructor(
        self, name: ConstructorFixtureName, /
    ) -> Constructor[Incomplete] | None:
        """Return a `Constructor` if the backend supports fixture `name`.

        The returned instance is callable, and gives easy access to:

            Implementation
            Implementation.is_*()
            backend_version
            identifier  # [<this-guy>] inside a `parametrize` id
        """
        if self.supports[name]:
            # TODO @dangotbanned: Probably should type `method`
            # since it can only be 1/3 return types (`ConstructorFixtureName`)
            method = getattr(self, name)
            return Constructor(method, name, self.identifier, self.implementation)
        return None

    def __init_subclass__(
        cls, *args: Any, import_or_skip_module: ModuleName | None = None, **kwds: Any
    ) -> None:
        super().__init_subclass__(*args, **kwds)
        if not (hasattr(cls, "backend_eager") or hasattr(cls, "backend_lazy")):
            msg = f"At least one of `backend_eager` or `backend_lazy` must be set as a class attribute for {cls!r}"
            raise TypeError(msg)
        if module := import_or_skip_module:
            cls.import_or_skip_module = module
        elif not (
            hasattr(cls, "import_or_skip_module")
            and (module := cls.import_or_skip_module)
        ):
            msg = f"`import_or_skip_module` is a required argument for direct subclasses of `EagerBackend`, got: {import_or_skip_module=} for {cls!r}"
            raise TypeError(msg)
        if cls.implementation is Implementation.UNKNOWN and cls.import_or_skip_module in {
            impl.value for impl in Implementation
        }:
            msg = (
                f"`{cls.import_or_skip_module=}` implies {cls!r} should use {Implementation(import_or_skip_module)!r},\n"
                f"but got: `{cls.implementation=}`"
            )
            raise TypeError(msg)

        cls.supports = backend_support_profile(cls)

        with _lock:
            TestBackend._BACKENDS[module].add(cls)

    @staticmethod
    def prepare_backends(
        *,
        include: OneOrIterable[Identifier] | Literal["ALL"] = "ALL",
        exclude: OneOrIterable[Identifier] | None = None,
    ) -> tuple[TestBackend[Any, Any, Any], ...]:
        """Initialize all known backends that are currently installed.

        Arguments:
            include: Backend(s) that should be selected if available.
            exclude: Backend(s) that should not be selected, has lower precedence than `include`.
        """
        with _lock:
            # NOTE: Although we're not mutating anything, we could run into issues
            # if the size of `_BACKENDS` changes during iteration
            known = TestBackend._BACKENDS
            installed = chain.from_iterable(
                (backend_tps for name, backend_tps in known.items() if find_spec(name))
            )
            selected = (tp() for tp in installed)
            if include != "ALL":
                including = _parse_identifiers(include)
                selected = (b for b in selected if b.identifier in including)
            if exclude is not None:
                excluding = _parse_identifiers(exclude)
                selected = (b for b in selected if b.identifier not in excluding)
            return tuple(sorted(selected, key=attrgetter("identifier")))


def backend_support_profile(backend: type[TestBackend[Any, Any, Any]]) -> SupportProfile:
    """Check `native_*` methods and return True for all overrides."""

    def _(name: str, /) -> bool:
        native = f"native_{name}"
        return getattr(backend, native) != getattr(TestBackend, native)

    return SupportProfile(
        lazyframe=_("lazyframe"), dataframe=_("dataframe"), series=_("series")
    )


class Constructor(Generic[R_co]):
    """Metadata-rich constructor wrapper.

    Fixtures wrapped in this way provide access to things you may need in `request.applymarker`.
    """

    __slots__ = ("_function", "fixture_name", "identifier", "implementation")
    _function: _Constructor[R_co]
    fixture_name: ConstructorFixtureName
    identifier: Identifier
    implementation: Implementation

    def __init__(
        self,
        bound_method: _Constructor[R_co],
        name: ConstructorFixtureName,
        identifier: BackendName | ModuleName | LiteralString,
        implementation: Implementation,
        /,
    ) -> None:
        self._function = bound_method
        self.fixture_name = name
        self.identifier = identifier
        self.implementation = implementation

    def __call__(self, data: Any, *args: Any, **kwds: Any) -> R_co:
        return self._function(data, *args, **kwds)

    def __repr__(self) -> str:
        return f"Constructor<{self.fixture_name}[{self.identifier}]>"

    def is_polars(self) -> bool:
        return self.implementation.is_polars()

    def is_pyarrow(self) -> bool:
        return self.implementation.is_pyarrow()

    def backend_version(self, *, unknown: UnknownBehavior = "ignore") -> tuple[int, ...]:
        version = self.implementation._backend_version()
        if (self.implementation is not Implementation.UNKNOWN) or unknown == "ignore":
            return version
        msg = f"TODO: Add support for {self.backend_version.__qualname__}({unknown=}) when integrating plugins\n{self!r}"
        raise NotImplementedError(msg)

    def xfail_not_implemented(
        self,
        request: pytest.FixtureRequest,
        /,
        condition: bool,  # noqa: FBT001
        method: LiteralString,
        *,
        raises: type[Exception] | tuple[type[Exception], ...] = NotImplementedError,
    ) -> None:
        request.applymarker(
            pytest.mark.xfail(
                condition,
                raises=raises,
                reason=f"TODO @dangotbanned: `{self.fixture_name}[{self.identifier}].{method}`",
            )
        )

    def xfail_polars_select(
        self,
        request: pytest.FixtureRequest,
        /,
        *,
        raises: type[Exception] | tuple[type[Exception], ...] = NotImplementedError,
    ) -> None:
        self.xfail_not_implemented(request, self.is_polars(), "select", raises=raises)

    def xfail_polars_with_columns(self, request: pytest.FixtureRequest, /) -> None:
        self.xfail_not_implemented(request, self.is_polars(), "with_columns")


LazyFrame: TypeAlias = Constructor[nwp.LazyFrame[Any]]
"""The type of the `lazyframe` fixture."""

DataFrame: TypeAlias = Constructor[nwp.DataFrame[Any, Any]]
"""The type of the `dataframe` fixture."""

Series: TypeAlias = Constructor[nwp.Series[Any]]
"""The type of the `series` fixture."""


class PolarsBackend(
    TestBackend["pl.LazyFrame", "pl.DataFrame", "pl.Series"],
    import_or_skip_module="polars",
):
    backend_eager = "polars"
    backend_lazy = "polars"
    implementation = Implementation.POLARS

    def schema_to_native(self, schema: Schema, /, **kwds: Any) -> pl.Schema:
        return schema.to_polars()

    def dtype_to_native(self, dtype: IntoDType, /, **kwds: Any) -> pl.DataType:
        from narwhals._plan.polars.namespace import dtype_to_native

        return dtype_to_native(dtype, Version.MAIN)

    def native_lazyframe(
        self, data: Mapping[str, Any], /, *, schema: Schema | None = None, **kwds: Any
    ) -> pl.LazyFrame:
        import polars as pl

        return pl.LazyFrame(data, self.schema_to_native(schema) if schema else None)

    def native_dataframe(
        self, data: Mapping[str, Any], /, *, schema: Schema | None = None, **kwds: Any
    ) -> pl.DataFrame:
        import polars as pl

        return pl.DataFrame(data, self.schema_to_native(schema) if schema else None)

    def native_series(
        self, values: Iterable[Any], /, *, dtype: IntoDType | None = None, **kwds: Any
    ) -> pl.Series:
        import polars as pl

        # NOTE: polars accepts a lot of specific types, + Sequence, Arrow streams, unsized iterables
        # but doesn't declare `Iterable` support
        data: Incomplete = values
        return pl.Series(data, dtype=(self.dtype_to_native(dtype) if dtype else None))


class ArrowBackend(
    TestBackend[Incomplete, "pa.Table", "pa.ChunkedArray[Any]"],
    import_or_skip_module="pyarrow",
):
    backend_eager = "pyarrow"
    implementation = Implementation.PYARROW

    def schema_to_native(self, schema: Schema, /, **kwds: Any) -> pa.Schema:
        return schema.to_arrow()

    def dtype_to_native(self, dtype: IntoDType, /, **kwds: Any) -> pa.DataType:
        from narwhals._plan.arrow.functions import dtype_native

        return dtype_native(dtype, Version.MAIN)

    def native_dataframe(
        self, data: Mapping[str, Any], /, *, schema: Schema | None = None, **kwds: Any
    ) -> pa.Table:
        import pyarrow as pa

        return pa.Table.from_pydict(
            data, self.schema_to_native(schema) if schema else None
        )

    def native_series(
        self, values: Iterable[Any], /, *, dtype: IntoDType | None = None, **kwds: Any
    ) -> pa.ChunkedArray[Any]:
        import pyarrow as pa

        # NOTE: The stubs crash language servers due to how many overlapping overloads this hits
        array: Incomplete = pa.chunked_array
        return array([values], self.dtype_to_native(dtype) if dtype else None)


# TODO @dangotbanned: Make this a fixture, move to `tests.plan.conftest.py`
def dataframe(
    data: Mapping[str, Any], /
) -> nwp.DataFrame[pa.Table, pa.ChunkedArray[Any]]:
    return nwp.DataFrame.from_native(pa.Table.from_pydict(data))


# TODO @dangotbanned: Make this a fixture, move to `tests.plan.conftest.py`
def series(values: Iterable[Any], /) -> nwp.Series[pa.ChunkedArray[Any]]:
    return nwp.Series.from_native(pa.chunked_array([values]))


def assert_equal_data(
    result: nwp.DataFrame[Any, Any], expected: Mapping[str, Any] | nwp.DataFrame[Any, Any]
) -> None:
    if isinstance(expected, nwp.DataFrame):
        expected = expected.to_dict(as_series=False)
    _assert_equal_data(result.to_dict(as_series=False), expected)


@overload
def assert_equal_series(result: nwp.Series[Any], expected: nwp.Series[Any]) -> None: ...
@overload
def assert_equal_series(
    result: nwp.Series[Any], expected: Iterable[Any], name: str
) -> None: ...
def assert_equal_series(
    result: nwp.Series[Any], expected: Iterable[Any], name: str = ""
) -> None:
    if isinstance(expected, nwp.Series):
        name = expected.name
        expected = expected.to_list()
    else:
        expected = expected if isinstance(expected, Sequence) else tuple(expected)
    assert_equal_data(result.to_frame(), {name: expected})


def re_compile(
    pattern: str, flags: _Flags = re.DOTALL | re.IGNORECASE
) -> re.Pattern[str]:
    """Compile a regular expression pattern, returning a Pattern object.

    Helper to default to using `flags=re.DOTALL | re.IGNORECASE`.
    """
    return re.compile(pattern, flags)
