from __future__ import annotations

import re
import sys
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Iterator
from typing import Mapping
from typing import Sequence

from narwhals._compliant.typing import CompliantDataFrameT_co
from narwhals._compliant.typing import CompliantExprAny
from narwhals._compliant.typing import CompliantExprT_contra
from narwhals._compliant.typing import CompliantFrameT_co
from narwhals._expression_parsing import is_elementary_expression
from narwhals._translate import TypeVar  # type: ignore[attr-defined]

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

NativeAggregationT_co = TypeVar("NativeAggregationT_co", covariant=True, default="str")
"""Some backends *may* return a `Callable` instead of a `str` referring to one."""


UNNAMED_PATTERN: re.Pattern[str] = re.compile(r"(\w+->)")
"""I'm unsure what this should be called.

Seems to be used as a way to get `thing_n`:

    "thing_1->thing_2->...->thing_n"

But with the assumption that `depth` is constrained below `2` (maybe?).

**In isolation - the pattern doesn't mean any of that.** 🤔
"""


class CompliantGroupBy(
    Protocol38[CompliantFrameT_co, CompliantExprT_contra, NativeAggregationT_co]
):
    _NARWHALS_TO_NATIVE_AGGREGATIONS: ClassVar[Mapping[str, Any]]
    _compliant_frame: Any
    _keys: Sequence[str]

    def __init__(
        self,
        compliant_frame: CompliantFrameT_co,
        keys: Sequence[str],
        /,
        *,
        drop_null_keys: bool,
    ) -> None: ...
    @property
    def compliant(self) -> CompliantFrameT_co:
        return self._compliant_frame  # type: ignore[no-any-return]

    def agg(self, *exprs: CompliantExprT_contra) -> CompliantFrameT_co: ...

    def _ensure_all_simple(self, exprs: Sequence[CompliantExprT_contra]) -> None:
        for expr in exprs:
            if not self._is_simple(expr):
                name = self.compliant._implementation.name.lower()
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

    @classmethod
    def _is_simple(cls, expr: CompliantExprAny, /) -> bool:
        """Return `True` is we can efficiently use `expr` in a native `group_by` context."""
        return (
            is_elementary_expression(expr)
            and cls._leaf_name(expr) in cls._NARWHALS_TO_NATIVE_AGGREGATIONS
        )

    @classmethod
    def _remap_expr_name(cls, name: str, /) -> NativeAggregationT_co:
        """Replace `name`, with some native representation.

        Arguments:
            name: Name of a `nw.Expr` aggregation method.

        Returns:
            A native compatible representation.
        """
        return cls._NARWHALS_TO_NATIVE_AGGREGATIONS.get(name, name)

    @classmethod
    def _leaf_name(cls, expr: CompliantExprAny, /) -> str:
        """Return the last function name in the chain defined by `expr`."""
        return UNNAMED_PATTERN.sub("", expr._function_name)


class EagerGroupBy(
    CompliantGroupBy[CompliantDataFrameT_co, CompliantExprT_contra, str],
    Protocol38[CompliantDataFrameT_co, CompliantExprT_contra],
):
    def __iter__(self) -> Iterator[tuple[Any, CompliantDataFrameT_co]]: ...
