from typing import TYPE_CHECKING
from typing import NoReturn
from typing import TypeVar
from typing import Union

if TYPE_CHECKING:
    import sys

    if sys.version_info >= (3, 10):
        from typing import TypeAlias
    else:
        from typing_extensions import TypeAlias

    from narwhals.expression import Expr
    from narwhals.series import Series

    IntoExpr: TypeAlias = Union[Expr, str, int, float, Series]

    NativeDataFrame = TypeVar("NativeDataFrame")
    NativeSeries = TypeVar("NativeSeries")


def assert_never(_: NoReturn) -> NoReturn:
    raise AssertionError("Expected code to be unreachable")
