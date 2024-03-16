from typing import TYPE_CHECKING
from typing import TypeAlias
from typing import TypeVar

if TYPE_CHECKING:
    from narwhals.expression import Expr
    from narwhals.series import Series

    IntoExpr: TypeAlias = Expr | str | int | float | Series

    NativeDataFrame = TypeVar("NativeDataFrame")
    NativeSeries = TypeVar("NativeSeries")

T = TypeVar("T")
