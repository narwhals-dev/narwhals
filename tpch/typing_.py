from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeVar

import narwhals as nw

if TYPE_CHECKING:
    from typing_extensions import ParamSpec, TypeAlias

    P = ParamSpec("P")
    R = TypeVar("R")


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
DataLoader = Callable[[QueryID], tuple[nw.LazyFrame[Any], ...]]


class QueryModule(Protocol):
    def query(
        self, *args: nw.LazyFrame[Any], **kwds: nw.LazyFrame[Any]
    ) -> nw.LazyFrame[Any]: ...


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
