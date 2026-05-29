"""Flags for features not available in all supported `polars` versions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final, Literal, TypeAlias, TypedDict

from narwhals._polars.utils import (
    SERIES_ACCEPTS_PD_INDEX as SERIES_ACCEPTS_PD_INDEX,  # noqa: PLC0414
    SERIES_RESPECTS_DTYPE as SERIES_RESPECTS_DTYPE,  # noqa: PLC0414
)
from narwhals._utils import Implementation

if TYPE_CHECKING:
    from collections.abc import Mapping, Sized

    from typing_extensions import LiteralString

    from narwhals._plan.options import ExplodeOptions, SortMultipleOptions
    from narwhals._plan.typing import Seq

    # TODO @dangotbanned: Replace with `dict[Literal["descending", "nulls_last"], Seq[bool]]` after bumping mypy
    # to include https://github.com/python/mypy/pull/20416
    class _SortOptions(TypedDict):
        descending: bool | Seq[bool]
        nulls_last: bool | Seq[bool]

    class _ExplodeOptions(TypedDict, total=False):
        empty_as_null: bool
        keep_nulls: bool


BACKEND_VERSION: Final = Implementation.POLARS._backend_version()
"""Static backend version for `polars`."""

# TODO @dangotbanned: Remove this dead code path on main, it is fixed at our minimum
HAS_POLARS_ERROR: Final = BACKEND_VERSION >= (0, 20, 4)
"""https://github.com/pola-rs/polars/pull/13615"""

HAS_LEN: Final = BACKEND_VERSION >= (0, 20, 5)
"""https://github.com/pola-rs/polars/pull/13719"""

SERIES_SORT_SUPPORTS_NULLS_LAST: Final = BACKEND_VERSION >= (0, 20, 6)
"""https://github.com/pola-rs/polars/pull/13794

Prior this this version, `nulls_last` was [only available on `*Frame` and `Expr`].

[only available on `*Frame` and `Expr`]: https://github.com/pola-rs/polars/issues/13788
"""

CONCAT_STR_SUPPORTS_IGNORE_NULLS: Final = BACKEND_VERSION >= (0, 20, 6)
"""https://github.com/pola-rs/polars/pull/13877"""


HAS_MEAN_HORIZONTAL: Final = BACKEND_VERSION >= (0, 20, 8)
"""https://github.com/pola-rs/polars/pull/14369"""

DUNDER_ARRAY_SUPPORTS_COPY: Final = BACKEND_VERSION >= (0, 20, 29)
"""https://github.com/pola-rs/polars/pull/16401"""

JOIN_OUTER_RENAMED_TO_FULL: Final = BACKEND_VERSION >= (0, 20, 29)
"""https://github.com/pola-rs/polars/pull/16417"""

SERIES_HAS_HAS_NULLS: Final = BACKEND_VERSION >= (0, 20, 30)
"""https://github.com/pola-rs/polars/pull/16488"""

MAP_BATCHES_SUPPORTS_RETURNS_SCALAR: Final = BACKEND_VERSION >= (0, 20, 31)
"""https://github.com/pola-rs/polars/pull/16556"""

NULLS_LAST_ACCEPTS_MULTIPLE: Final = BACKEND_VERSION >= (0, 20, 31)
"""https://github.com/pola-rs/polars/pull/16639"""

EWM_MEAN_PRESERVES_NULLS: Final = BACKEND_VERSION >= (1,)
"""https://github.com/pola-rs/polars/pull/15503"""

MELT_RENAMED_TO_UNPIVOT: Final = BACKEND_VERSION >= (1, 0)
"""https://github.com/pola-rs/polars/pull/17095"""

LAZYFRAME_HAS_COLLECT_SCHEMA: Final = BACKEND_VERSION >= (1, 0)

HAS_REPLACE_STRICT: Final = BACKEND_VERSION >= (1, 0)
"""https://github.com/pola-rs/polars/pull/16921

`replace` had two parameters deprecated here.
"""

ROLLING_VAR_STD_STABLE: Final = BACKEND_VERSION >= (1, 0)
"""https://github.com/narwhals-dev/narwhals/pull/1451#issuecomment-2506066114"""

CONSTRUCTOR_ACCEPTS_PYCAPSULE: Final = BACKEND_VERSION >= (1, 3)
"""https://github.com/pola-rs/polars/pull/17693"""

SERIES_HAS_FIRST_LAST: Final = BACKEND_VERSION >= (1, 10)
"""https://github.com/pola-rs/polars/pull/19093"""

SERIES_RFLOORDIV_HANDLES_ZERO: Final = BACKEND_VERSION >= (1, 10)
"""https://github.com/pola-rs/polars/issues/19142

Note:
    The bug impacts `__rmod__` as well, but didn't get fixed in narwhals?
"""

LIT_ACCEPTS_DICT: Final = BACKEND_VERSION >= (1, 10)
"""https://github.com/pola-rs/polars/pull/19214"""

SERIES_RPOW_PRESERVES_NAME: Final = BACKEND_VERSION >= (1, 16, 1)
"""https://github.com/pola-rs/polars/pull/20072"""

IS_NAN_FINITE_NUMERIC_PRESERVES_NULLS: Final = BACKEND_VERSION >= (1, 18)
"""https://github.com/pola-rs/polars/pull/20386"""

MIN_PERIODS_RENAMED_TO_MIN_SAMPLES: Final = BACKEND_VERSION >= (1, 21)
"""https://github.com/pola-rs/polars/pull/20850"""

HAS_LINEAR_SPACE: Final = BACKEND_VERSION >= (1, 21)
"""https://github.com/pola-rs/polars/pull/20678"""

HAS_DATA_TYPE_EXPR: Final = BACKEND_VERSION >= (1, 32)
"""[#23167], [#23249], [#23353].

Limited support starts at `1.31.0`, main changes follow in `1.32.0`.

[#23167]: https://github.com/pola-rs/polars/pull/23167
[#23249]: https://github.com/pola-rs/polars/pull/23249
[#23353]: https://github.com/pola-rs/polars/pull/23353
"""

PIVOT_SUPPORTS_ON_COLUMNS: Final = BACKEND_VERSION >= (1, 36)
"""https://github.com/pola-rs/polars/pull/25016"""

EXPLODE_SUPPORTS_OPTIONS: Final = BACKEND_VERSION >= (1, 36)
"""[#25289], [#25369]

[#25289]: https://github.com/pola-rs/polars/pull/25289
[#25369]: https://github.com/pola-rs/polars/pull/25369
"""

# TODO @dangotbanned: Use in `Expr.is_in`
NO_LIST_AS_SERIES_IS_IN: Final = BACKEND_VERSION >= (1, 28)
"""[#22149], [#22178]

[#22149]: https://github.com/pola-rs/polars/issues/22149
[#22178]: https://github.com/pola-rs/polars/pull/22178
"""

NO_LIST_AS_SERIES_REPLACE_STRICT: Final = BACKEND_VERSION >= (1, 29)
"""[#22149], [#22465]

Also, `pl.lit(tuple)` behaves differently *prior to* these changes

[#22149]: https://github.com/pola-rs/polars/issues/22149
[#22465]: https://github.com/pola-rs/polars/pull/22465
"""

OVER_SUPPORTS_ORDER_BY: Final = BACKEND_VERSION >= (1, 10)
"""Added in [#16743], fixed in the version after [#18947].

Supports:

    pl.col("a").first().over("b", order_by="c")

## Notes
- It is unclear exactly which expressions were not valid before `1.10`
  - length-preserving functions are the most common issue
- Some bugs look like they're data-dependent
  - There's a chunk of test suite that *would* pass,
    but we can't allow the general case where incorrect results occur

[#16743]: https://github.com/pola-rs/polars/pull/16743
[#18947]: https://github.com/pola-rs/polars/pull/18947
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
"""[#22712]

Supports:

    pl.col("a").first().over(order_by="c", descending=True, nulls_last=False)


Until [#24874] (`>=1.35.0`), the workaround of `pl.lit(1)` does not work in all contexts.

So, `pl.repeat(1, len())` is required instead.

[#22712]: https://github.com/pola-rs/polars/pull/22712
[#24874]: https://github.com/pola-rs/polars/pull/24874
"""

OVER_RESPECTS_NULLS_LAST: Final = BACKEND_VERSION >= (1, 39)
"""https://github.com/pola-rs/polars/pull/26718

Supports:

    pl.col("a").first().over(order_by="c", descending=True, nulls_last=True)
"""

_OverFeature: TypeAlias = Literal["order_by_any", "descending", "nulls_last"]

_OVER_ERRORS: Final[Mapping[_OverFeature, tuple[str, str]]] = {
    "order_by_any": ("..., order_by=...", "1.10.0"),
    "descending": ("..., order_by=..., descending=True", "1.22.0"),
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

    def sort(self: SortMultipleOptions, by: Sized | int, /) -> _SortOptions:
        """Try to convert `self` into something the current version of polars supports.

        [`extend_bool`] doesn't broadcast length 1 sequences, so we do it here.

        [`extend_bool`]: https://github.com/pola-rs/polars/blob/b8bfb07a4a37a8d449d6d1841e345817431142df/py-polars/polars/_utils/various.py#L580-L594
        """
        desc, nulls = self.descending, self.nulls_last
        len_by = by if isinstance(by, int) else len(by)
        if len_by != 1:
            desc = desc if len(desc) != 1 else desc * len_by
            nulls = nulls if len(nulls) != 1 else nulls * len_by
        return {"descending": desc, "nulls_last": nulls}
else:

    def sort(self: SortMultipleOptions, by: Sized | int, /) -> _SortOptions:
        desc, nulls = self.descending, self.nulls_last
        desc = desc if len(desc) != 1 else desc * (by if isinstance(by, int) else len(by))
        first = nulls[0]
        if len(nulls) != 1 and any(x != first for x in nulls[1:]):
            msg = "nulls_last=(..., )"
            raise too_old(msg, "0.20.31")
        return {"descending": desc, "nulls_last": first}


if EXPLODE_SUPPORTS_OPTIONS:

    def explode(
        self: ExplodeOptions | None = None,
        *,
        empty_as_null: bool = True,
        keep_nulls: bool = True,
    ) -> _ExplodeOptions:
        if self:
            empty_as_null, keep_nulls = self.empty_as_null, self.keep_nulls
        return {"empty_as_null": empty_as_null, "keep_nulls": keep_nulls}

else:
    # NOTE: The default is backwards compatible
    def explode(
        self: ExplodeOptions | None = None,
        *,
        empty_as_null: bool = True,
        keep_nulls: bool = True,
    ) -> _ExplodeOptions:
        if self:
            empty_as_null, keep_nulls = self.empty_as_null, self.keep_nulls
        if not (empty_as_null and keep_nulls):
            msg = f"explode({empty_as_null=}, {keep_nulls=})"
            raise too_old(msg, "1.36.0")  # pyright: ignore[reportArgumentType]
        return {}


_MIN_SAMPLES = "min_samples" if MIN_PERIODS_RENAMED_TO_MIN_SAMPLES else "min_periods"


def min_samples_periods(min_samples: int, **kwds: Any) -> dict[str, Any]:
    return {_MIN_SAMPLES: min_samples, **kwds}
