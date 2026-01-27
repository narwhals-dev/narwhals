from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeVar

from narwhals._typing import _EagerAllowedImpl, _LazyAllowedImpl

if TYPE_CHECKING:
    from collections.abc import Callable

    from typing_extensions import ParamSpec, TypeAlias

    import narwhals as nw
    from tpch.tests.conftest import Backend

    P = ParamSpec("P")
    R = TypeVar("R")

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


class QueryModule(Protocol):
    def query(
        self, *args: nw.LazyFrame[Any], **kwds: nw.LazyFrame[Any]
    ) -> nw.LazyFrame[Any]: ...


# TODO @dangotbanned: Rename this, it is used for `pytest.skip` too
class AssertExpected(Protocol):
    """Failure-state-context callback.

    The returned value will be passed to `pytest.mark.xfail(condition=...)`.
    """

    def __call__(self, backend: Backend, scale_factor: float, /) -> bool: ...


# TODO @dangotbanned: Remove
def todo_mark(*_: Any, **__: Any) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """TODO: Do something useful, not just a placeholder.

    - Run a callback to validate the known issue was the problem?
      - Would only want to run *after* failing, since it needs to be eager
    - Add pytest stuff?
    """

    def decorate(function: Callable[P, R], /) -> Callable[P, R]:
        @wraps(function)
        def wrapper(*args: P.args, **kwds: P.kwargs) -> R:
            return function(*args, **kwds)

        return wrapper

    return decorate
