from __future__ import annotations

import re
import sys
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Iterator
from typing import Mapping
from typing import Sequence

from narwhals._compliant.typing import CompliantDataFrameT
from narwhals._compliant.typing import CompliantExprT
from narwhals._compliant.typing import CompliantFrameT
from narwhals._compliant.typing import NativeFrameT_co
from narwhals._expression_parsing import is_elementary_expression

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

    from narwhals._compliant.dataframe import CompliantDataFrame
    from narwhals._compliant.dataframe import CompliantLazyFrame

    Frame: TypeAlias = "CompliantDataFrame[Any, Any, NativeFrameT_co] | CompliantLazyFrame[Any, NativeFrameT_co]"

if not TYPE_CHECKING:  # pragma: no cover
    if sys.version_info >= (3, 9):
        from typing import Protocol as Protocol38
    else:
        from typing import Generic as Protocol38
else:  # pragma: no cover
    # TODO @dangotbanned: Remove after dropping `3.8` (#2084)
    # - https://github.com/narwhals-dev/narwhals/pull/2064#discussion_r1965921386
    from typing import Protocol as Protocol38

__all__ = ["CompliantGroupBy", "EagerGroupBy"]


# NOTE: Type checkers disagree
# - `pyright` wants invariant `*Expr`
# - `mypy` want contravariant `*Expr`
class CompliantGroupBy(Protocol38[CompliantFrameT, CompliantExprT]):  # type: ignore[misc]
    _NARWHALS_TO_NATIVE_AGGREGATIONS: ClassVar[Mapping[str, Any]]
    _compliant_frame: CompliantFrameT
    _keys: Sequence[str]

    def __init__(
        self,
        compliant_frame: CompliantFrameT,
        keys: Sequence[str],
        *,
        drop_null_keys: bool,
    ) -> None: ...
    @property
    def compliant(self) -> CompliantFrameT:
        return self._compliant_frame

    @property
    def native(
        self: CompliantGroupBy[Frame[NativeFrameT_co], CompliantExprT],
    ) -> NativeFrameT_co:
        return self.compliant.native

    def agg(self, *exprs: CompliantExprT) -> CompliantFrameT: ...

    def _ensure_all_simple(self, exprs: Sequence[CompliantExprT]) -> None:
        for expr in exprs:
            if (
                not is_elementary_expression(expr)
                and re.sub(r"(\w+->)", "", expr._function_name)
                in self._NARWHALS_TO_NATIVE_AGGREGATIONS
            ):
                # NOTE: Need to define `_implementation` in both protocols (#2251)
                name = self.compliant._implementation.name.lower()  # type: ignore  # noqa: PGH003
                msg = (
                    f"Non-trivial complex aggregation found.\n\n"
                    f"Hint: you were probably trying to apply a non-elementary aggregation with a"
                    f"{name!r} table.\n"
                    "Please rewrite your query such that group-by aggregations "
                    "are elementary. For example, instead of:\n\n"
                    "    df.group_by('a').agg(nw.col('b').round(2).mean())\n\n"
                    "use:\n\n"
                    "    df.with_columns(nw.col('b').round(2)).group_by('a').agg(nw.col('b').mean())\n\n"
                )
                raise ValueError(msg)


class EagerGroupBy(  # type: ignore[misc]
    CompliantGroupBy[CompliantDataFrameT, CompliantExprT],
    Protocol38[CompliantDataFrameT, CompliantExprT],
):
    def __iter__(self) -> Iterator[tuple[Any, CompliantDataFrameT]]: ...
