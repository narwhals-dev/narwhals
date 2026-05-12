"""Flags for features not available in all supported `polars` versions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Literal, TypeAlias, TypedDict

from narwhals._polars.utils import (
    SERIES_ACCEPTS_PD_INDEX as SERIES_ACCEPTS_PD_INDEX,  # noqa: PLC0414
    SERIES_RESPECTS_DTYPE as SERIES_RESPECTS_DTYPE,  # noqa: PLC0414
)
from narwhals._utils import Implementation

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from typing_extensions import LiteralString

    from narwhals._plan.options import SortMultipleOptions
    from narwhals._plan.typing import Seq

    # TODO @dangotbanned: Replace with `dict[Literal["descending", "nulls_last"], Seq[bool]]` after bumping mypy
    # to include https://github.com/python/mypy/pull/20416
    class _SortOptions(TypedDict):
        descending: bool | Seq[bool]
        nulls_last: bool | Seq[bool]


BACKEND_VERSION: Final = Implementation.POLARS._backend_version()
"""Static backend version for `polars`."""

# TODO @dangotbanned: Remove this dead code path on main, it is fixed at our minimum
HAS_POLARS_ERROR: Final = BACKEND_VERSION >= (0, 20, 4)
"""https://github.com/pola-rs/polars/pull/13615"""


SERIES_SORT_SUPPORTS_NULLS_LAST: Final = BACKEND_VERSION >= (0, 20, 6)
"""https://github.com/pola-rs/polars/pull/13794

Prior this this version, `nulls_last` was [only available on `*Frame` and `Expr`].

[only available on `*Frame` and `Expr`]: https://github.com/pola-rs/polars/issues/13788
"""

DUNDER_ARRAY_SUPPORTS_COPY: Final = BACKEND_VERSION >= (0, 20, 29)
"""https://github.com/pola-rs/polars/pull/16401"""

JOIN_OUTER_RENAMED_TO_FULL: Final = BACKEND_VERSION >= (0, 20, 29)
"""https://github.com/pola-rs/polars/pull/16417"""

SERIES_HAS_HAS_NULLS: Final = BACKEND_VERSION >= (0, 20, 30)
"""https://github.com/pola-rs/polars/pull/16488"""

NULLS_LAST_ACCEPTS_MULTIPLE: Final = BACKEND_VERSION >= (0, 20, 31)
"""https://github.com/pola-rs/polars/pull/16639"""

MELT_RENAMED_TO_UNPIVOT: Final = BACKEND_VERSION >= (1, 0)
"""https://github.com/pola-rs/polars/pull/17095"""

LAZYFRAME_HAS_COLLECT_SCHEMA: Final = BACKEND_VERSION >= (1, 0)

CONSTRUCTOR_ACCEPTS_PYCAPSULE: Final = BACKEND_VERSION >= (1, 3)
"""https://github.com/pola-rs/polars/pull/17693"""

SERIES_HAS_FIRST_LAST: Final = BACKEND_VERSION >= (1, 10)
"""https://github.com/pola-rs/polars/pull/19093"""

SERIES_RFLOORDIV_HANDLES_ZERO: Final = BACKEND_VERSION >= (1, 10)
"""https://github.com/pola-rs/polars/issues/19142

Note:
    The bug impacts `__rmod__` as well, but didn't get fixed in narwhals?
"""

SERIES_RPOW_PRESERVES_NAME: Final = BACKEND_VERSION >= (1, 16, 1)
"""https://github.com/pola-rs/polars/pull/20072"""

IS_NAN_NUMERIC_PROPAGATES_NULLS: Final = BACKEND_VERSION >= (1, 18)
"""https://github.com/pola-rs/polars/pull/20386"""

MIN_PERIODS_RENAMED_TO_MIN_SAMPLES: Final = BACKEND_VERSION >= (1, 21)
"""https://github.com/pola-rs/polars/pull/20850"""

HAS_LINEAR_SPACE: Final = BACKEND_VERSION >= (1, 21)
"""https://github.com/pola-rs/polars/pull/20678"""

PIVOT_SUPPORTS_ON_COLUMNS: Final = BACKEND_VERSION >= (1, 36)
"""https://github.com/pola-rs/polars/pull/25016"""


OVER_SUPPORTS_ORDER_BY: Final = BACKEND_VERSION >= (1, 0)
"""https://github.com/pola-rs/polars/pull/16743

Supports:

    pl.col("a").first().over("b", order_by="c")
"""

OVER_ORDER_BY_GROUP_BY_FIX: Final = BACKEND_VERSION >= (1, 9)
"""https://github.com/MarcoGorelli/narwhals/commit/5f35d22857f231bc0fba6ace6389cb58d742923c

- The condition checks for `< 1.9`
- The message says         `<= 1.9`
- Only thing for `over` in that release is https://github.com/pola-rs/polars/pull/18947
"""

OVER_SUPPORTS_DESCENDING: Final = BACKEND_VERSION >= (1, 22)
"""[#20919], [#20952]

Supports:

    pl.col("a").first().over("b", order_by="c", descending=True)
    pl.col("a").first().over("b", order_by="c", descending=True, nulls_last=False)

[#20919]: https://github.com/pola-rs/polars/pull/20919
[#20952]: https://github.com/pola-rs/polars/pull/20952
"""

OVER_WITHOUT_PARTITION_BY: Final = BACKEND_VERSION >= (1, 30)
"""https://github.com/pola-rs/polars/pull/22712

Supports:

    pl.col("a").first().over(order_by="c", descending=True, nulls_last=False)
"""

OVER_RESPECTS_NULLS_LAST: Final = BACKEND_VERSION >= (1, 39)
"""https://github.com/pola-rs/polars/pull/26718

Supports:

    pl.col("a").first().over(order_by="c", descending=True, nulls_last=True)
"""

_OverFeature: TypeAlias = Literal[
    "order_by_any", "descending", "order_by_only", "nulls_last"
]

_OVER_ERRORS: Final[Mapping[_OverFeature, tuple[str, str]]] = {
    "order_by_any": ("..., order_by=...", "1.0.0"),
    "descending": ("..., order_by=..., descending=True", "1.22.0"),
    "order_by_only": ("order_by=...", "1.30.0"),
    "nulls_last": ("order_by=..., nulls_last=True", "1.39.0"),
}


def over_error(feature: _OverFeature, /) -> NotImplementedError:
    args, version = _OVER_ERRORS[feature]
    return too_old(f"over({args})", version)


def too_old(code: LiteralString, version: LiteralString, /) -> NotImplementedError:
    """Create an error for a version of polars that's `too_old`.

    >>> too_old("DataFrame.pivot(...)", "1.0.0")
    NotImplementedError('`DataFrame.pivot(...)` requires `polars>=1.0.0`')

    Tip:
        Consider adding a layer above this for anything with complexity (see `over_error`)
    """
    return NotImplementedError(f"`{code}` requires `polars>={version}`")


if NULLS_LAST_ACCEPTS_MULTIPLE:

    def sort(self: SortMultipleOptions, by: Sequence[str], /) -> _SortOptions:
        """Try to convert `self` into something the current version of polars supports.

        [`extend_bool`] doesn't broadcast length 1 sequences, so we do it here.

        [`extend_bool`]: https://github.com/pola-rs/polars/blob/b8bfb07a4a37a8d449d6d1841e345817431142df/py-polars/polars/_utils/various.py#L580-L594
        """
        desc, nulls = self.descending, self.nulls_last
        len_by = len(by)
        if len_by != 1:
            desc = desc if len(desc) != 1 else desc * len_by
            nulls = nulls if len(nulls) != 1 else nulls * len_by
        return {"descending": desc, "nulls_last": nulls}
else:

    def sort(self: SortMultipleOptions, by: Sequence[str], /) -> _SortOptions:
        desc, nulls = self.descending, self.nulls_last
        desc = desc if len(desc) != 1 else desc * len(by)
        first = nulls[0]
        if len(nulls) != 1 and any(x != first for x in nulls[1:]):
            msg = "nulls_last=(..., )"
            raise too_old(msg, "0.20.31")
        return {"descending": desc, "nulls_last": first}
