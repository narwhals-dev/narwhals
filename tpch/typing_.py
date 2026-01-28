from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Protocol

from narwhals._typing import _EagerAllowedImpl, _LazyAllowedImpl

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

    import narwhals as nw
    from tpch.classes import Backend


KnownImpl: TypeAlias = Literal[_EagerAllowedImpl, _LazyAllowedImpl]
TPCHBackend: TypeAlias = Literal[
    "polars[lazy]", "pyarrow", "pandas[pyarrow]", "dask", "duckdb", "sqlframe"
]
QueryID: TypeAlias = Literal[
    "q1",
    "q2",
    "q3",
    "q4",
    "q5",
    "q6",
    "q7",
    "q8",
    "q9",
    "q10",
    "q11",
    "q12",
    "q13",
    "q14",
    "q15",
    "q16",
    "q17",
    "q18",
    "q19",
    "q20",
    "q21",
    "q22",
]
XFailRaises: TypeAlias = type[BaseException] | tuple[type[BaseException], ...]
BuiltinScaleFactor: TypeAlias = Literal["0.01", "0.1", "1.0"]
Artifact: TypeAlias = Literal["database", "answers"]


class QueryModule(Protocol):
    def query(
        self, *args: nw.LazyFrame[Any], **kwds: nw.LazyFrame[Any]
    ) -> nw.LazyFrame[Any]: ...


class Predicate(Protocol):
    """Failure-state-context callback.

    The returned value will be used in either:

        pytest.mark.xfail(predicate(backend, scale_factor))

    Or:

        if predicate(backend, scale_factor):
            pytest.skip()
    """

    def __call__(self, backend: Backend, scale_factor: float, /) -> bool: ...
