from typing import TYPE_CHECKING  # pragma: no cover
from typing import TypeVar  # pragma: no cover

if TYPE_CHECKING:
    from typing import TypeAlias

    from narwhals.expression import Expr
    from narwhals.series import Series

    IntoExpr: TypeAlias = Expr | str | int | float | Series

    NativeDataFrame = TypeVar("NativeDataFrame")
    NativeSeries = TypeVar("NativeSeries")
