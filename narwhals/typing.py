from typing import TYPE_CHECKING  # pragma: no cover
from typing import TypeVar  # pragma: no cover
from typing import Union  # pragma: no cover

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
