from __future__ import annotations  # pragma: no cover

from typing import TYPE_CHECKING  # pragma: no cover
from typing import Union  # pragma: no cover

if TYPE_CHECKING:
    import sys

    if sys.version_info >= (3, 10):
        from typing import TypeAlias
    else:
        from typing_extensions import TypeAlias

    from narwhals._spark.expr import PySparkExpr

    IntoPySparkExpr: TypeAlias = Union[PySparkExpr, str]
