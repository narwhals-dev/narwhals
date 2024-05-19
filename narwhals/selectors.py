from narwhals.dtypes import DType
from narwhals.dtypes import translate_dtype
from narwhals.expression import Expr
from narwhals.utils import flatten


def by_dtype(*dtypes: DType) -> Expr:
    return Expr(
        lambda plx: plx.selectors.by_dtype(
            [translate_dtype(plx, dtype) for dtype in flatten(dtypes)]
        )
    )
