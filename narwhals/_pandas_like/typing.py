from __future__ import annotations  # pragma: no cover

from typing import TYPE_CHECKING  # pragma: no cover
from typing import Union  # pragma: no cover

if TYPE_CHECKING:
    import sys
    from typing import Literal

    from typing_extensions import ReadOnly
    from typing_extensions import TypedDict

    if sys.version_info >= (3, 10):
        from typing import TypeAlias
    else:
        from typing_extensions import TypeAlias

    from narwhals._pandas_like.expr import PandasLikeExpr
    from narwhals._pandas_like.series import PandasLikeSeries

    IntoPandasLikeExpr: TypeAlias = Union[PandasLikeExpr, PandasLikeSeries]

    class AggregationsToPandasEquivalent(TypedDict):
        sum: ReadOnly[Literal["sum"]]
        mean: ReadOnly[Literal["mean"]]
        median: ReadOnly[Literal["median"]]
        max: ReadOnly[Literal["max"]]
        min: ReadOnly[Literal["min"]]
        std: ReadOnly[Literal["std"]]
        var: ReadOnly[Literal["var"]]
        len: ReadOnly[Literal["size"]]
        n_unique: ReadOnly[Literal["nunique"]]
        count: ReadOnly[Literal["count"]]

    class WindowFunctionsToPandasEquivalent(TypedDict):
        cum_sum: ReadOnly[Literal["cumsum"]]
        cum_min: ReadOnly[Literal["cummin"]]
        cum_max: ReadOnly[Literal["cummax"]]
        cum_prod: ReadOnly[Literal["cumprod"]]
        cum_count: ReadOnly[Literal["cumsum"]]
        shift: ReadOnly[Literal["shift"]]
        rank: ReadOnly[Literal["rank"]]
        diff: ReadOnly[Literal["diff"]]

    WindowFunctionName: TypeAlias = Literal[
        "cum_sum", "cum_min", "cum_max", "cum_prod", "cum_count", "shift", "rank", "diff"
    ]
