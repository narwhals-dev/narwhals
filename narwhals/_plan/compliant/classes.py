"""The class-accessing parts of `*Namespace`.

## Notes
- Separates the operations from navigation
- Exposed in the `Plugin`-level as `__narwhals_classes__`
- Intended to be used in a similar way for the implementation
    - But not by accessing through the plugin
- A stepping stone to the `Package` idea.
    - Everything is covariant already
    - This object can be put anywhere
    - Still uses the current names, so less noise to migrate
- Versioning
    - `__narwhals_classes__` gives you access to every version of the classes available (e.g `.v1`)
    - Each version is represented with 1 unique type parameter
        - The interface exposed here is fully covariant
        - Within a version, and implementation can use invariant types and not cause issues outside it
    - Supporting versions is not a requirement for `Plugin` but is for `Builtin`
        - TODO @dangotbanned: Find where either Marco or Francesco said that
    - Checking for version support is just checking if that property exists
        - E.g. do that once per-plugin and then it can be included/excluded for all classes
"""

from __future__ import annotations

# ruff: noqa: PLC0105
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, TypeVar, overload

from narwhals._plan.common import hasattrs_static
from narwhals._plan.compliant.typing import (
    # NOTE: The names are simply too long!
    DataFrameAny,
    DataFrameT_co as DF,
    EagerDataFrameAny,
    EagerDataFrameT_co as EagerDF,
    EagerExprAny,
    EagerExprT_co as EagerE,
    EagerScalarAny,
    EagerScalarNoDefaultT_co as EagerSC,
    ExprAny,
    ExprT_co as E,
    LazyFrameAny,
    LazyFrameT_co as LF,
    PlanEvaluatorT_co as PE,
    ScalarAny,
    ScalarNoDefaultT_co as SC,
    SeriesAny,
    SeriesT_co as S,
)

if TYPE_CHECKING:
    from typing_extensions import TypeAlias, TypeIs

    from narwhals._plan.plans.visitors import ResolvedToCompliantAny as PlanEvaluatorAny
    from narwhals._utils import Version


EagerAny: TypeAlias = (
    "EagerClasses[DataFrameAny, SeriesAny, ExprAny, ExprAny | ScalarAny]"
)
LazyAny: TypeAlias = (
    "LazyClasses[LazyFrameAny, PlanEvaluatorAny, ExprAny, ExprAny | ScalarAny]"
)
HybridAny: TypeAlias = "HybridClasses[DataFrameAny, SeriesAny, LazyFrameAny, PlanEvaluatorAny, ExprAny, ExprAny | ScalarAny]"

ClassesAny: TypeAlias = "EagerAny | LazyAny | HybridAny"
"""The type of either `__narwhals_classes__` or `__narwhals_classes__.v*`.

Can provide eager, lazy or a combination of the two.
"""

ClassesImplAny: TypeAlias = "ClassesAny | EagerImplAny"
"""Keep this separated from `Plugin`."""


C = TypeVar("C", bound=ClassesAny, covariant=True)
"""Covariant TypeVar for the `__narwhals_classes__` property."""

C1 = TypeVar("C1", bound=ClassesAny, covariant=True)
"""Covariant TypeVar for the `__narwhals_classes__.v1` property."""

C2 = TypeVar("C2", bound=ClassesAny, covariant=True)
"""Covariant TypeVar for the `__narwhals_classes__.v2` property."""

EagerImplC = TypeVar("EagerImplC", bound="EagerImplVAllAny", covariant=True)
EagerImplC1 = TypeVar("EagerImplC1", bound="EagerImplAny", covariant=True)
EagerImplC2 = TypeVar("EagerImplC2", bound="EagerImplAny", covariant=True)


class HasClasses(Protocol[C]):
    """An object or module that defines `__narwhals_classes__`."""

    __slots__ = ()

    @property
    def __narwhals_classes__(self) -> C:
        """The backend's compliant-level types."""
        ...


class HasV1(Protocol[C1]):
    """An object that defines `v1`."""

    __slots__ = ()

    @property
    def v1(self) -> C1:
        """The classes to use with `Version.V1`."""
        ...


class HasV2(Protocol[C2]):
    """An object that defines `v2`."""

    __slots__ = ()

    @property
    def v2(self) -> C2:
        """The classes to use with `Version.V2`."""
        ...


class HasVAll(HasV1[C1], HasV2[C2], Protocol[C1, C2]):
    """An object that supports all versions."""

    __slots__ = ()


# NOTE: The type of `__narwhals_classes__`, which defines a `v1` or `v2` property
# Eager
EagerV1Any: TypeAlias = (
    "EagerClassesV1[EagerAny, DataFrameAny, SeriesAny, ExprAny, ExprAny | ScalarAny]"
)
EagerV2Any: TypeAlias = (
    "EagerClassesV2[EagerAny, DataFrameAny, SeriesAny, ExprAny, ExprAny | ScalarAny]"
)
EagerVAllAny: TypeAlias = "EagerClassesVAll[EagerAny, EagerAny, DataFrameAny, SeriesAny, ExprAny, ExprAny | ScalarAny]"

# EagerImpl
EagerImplAny: TypeAlias = "EagerImplClasses[EagerDataFrameAny, SeriesAny, EagerExprAny, EagerExprAny | EagerScalarAny]"
EagerImplVAllAny: TypeAlias = "EagerImplClassesVAll[EagerImplAny, EagerImplAny, EagerDataFrameAny, SeriesAny, EagerExprAny, EagerExprAny | EagerScalarAny]"

# Lazy
LazyV1Any: TypeAlias = (
    "LazyClassesV1[LazyAny, LazyFrameAny, PlanEvaluatorAny, ExprAny, ExprAny | ScalarAny]"
)
LazyV2Any: TypeAlias = (
    "LazyClassesV2[LazyAny, LazyFrameAny, PlanEvaluatorAny, ExprAny, ExprAny | ScalarAny]"
)
LazyVAllAny: TypeAlias = "LazyClassesVAll[LazyAny, LazyAny, LazyFrameAny, PlanEvaluatorAny, ExprAny, ExprAny | ScalarAny]"

# Hybrid
HybridV1Any: TypeAlias = "HybridClassesV1[HybridAny, DataFrameAny, SeriesAny, LazyFrameAny, PlanEvaluatorAny, ExprAny, ExprAny | ScalarAny]"
HybridV2Any: TypeAlias = "HybridClassesV2[HybridAny, DataFrameAny, SeriesAny, LazyFrameAny, PlanEvaluatorAny, ExprAny, ExprAny | ScalarAny]"
HybridVAllAny: TypeAlias = "HybridClassesVAll[HybridAny, HybridAny, DataFrameAny, SeriesAny, LazyFrameAny, PlanEvaluatorAny, ExprAny, ExprAny | ScalarAny]"

# Throw them in a pot and stir
ClassesVAny: TypeAlias = (
    "EagerV1Any | LazyV1Any | HybridV1Any | EagerV2Any | LazyV2Any | HybridV2Any"
)
"""The type of `__narwhals_classes__` which defines at least one versioned property."""

ClassesVAllAny: TypeAlias = (
    "EagerVAllAny | LazyVAllAny | HybridVAllAny | EagerImplVAllAny"
)
"""The type of `__narwhals_classes__` which defines all versioned properties."""


CB = TypeVar("CB", bound=ClassesVAllAny, covariant=True)
"""Covariant TypeVar for the `__narwhals_classes__` property on a `Builtin`.

To ensure we maintain [backwards compatibility], every `Builtin` is **required** to implement
support for every stable version (in addition to main).

An external `Plugin` *can choose* to support versioning.

[backwards compatibility]: https://narwhals-dev.github.io/narwhals/backcompat/
"""

CB1 = TypeVar("CB1", bound=ClassesImplAny, covariant=True)
CB2 = TypeVar("CB2", bound=ClassesImplAny, covariant=True)


class CompliantClasses(Protocol[E, SC]):
    """The common class-accessing parts of `*Namespace`.

    These are guaranteed to be available in both eager and lazy.
    """

    __slots__ = ()
    # NOTE: `Implementation` is available in the plugin and in each class, no need for here too
    version: ClassVar[Version]

    @property
    def _expr(self) -> type[E]: ...
    @property
    def _scalar(self) -> type[SC]: ...
    def __narwhals_expr_prepare__(self) -> E:  # pragma: no cover
        # NOTE: still needed for disambiguating `Expr`, `Scalar` and `Namespace`
        tp = self._expr
        return tp.__new__(tp)


class EagerClasses(CompliantClasses[E, SC], Protocol[DF, S, E, SC]):
    """Extends `CompliantClasses` with a `DataFrame` and `Series`."""

    __slots__ = ()

    @property
    def _dataframe(self) -> type[DF]: ...
    @property
    def _series(self) -> type[S]: ...


class LazyClasses(CompliantClasses[E, SC], Protocol[LF, PE, E, SC]):
    """Extends `CompliantClasses` with a `LazyFrame` and `PlanEvaluator`."""

    __slots__ = ()

    @property
    def _lazyframe(self) -> type[LF]: ...
    @property
    def _evaluator(self) -> type[PE]:
        """Translate a `ResolvedPlan` into `CompliantLazyFrame` operations."""
        ...


class HybridClasses(
    EagerClasses[DF, S, E, SC], LazyClasses[LF, PE, E, SC], Protocol[DF, S, LF, PE, E, SC]
):
    """Supports both eager and lazy operations."""

    __slots__ = ()


class EagerImplClasses(
    EagerClasses[EagerDF, S, EagerE, EagerSC], Protocol[EagerDF, S, EagerE, EagerSC]
):
    """This is what `pyarrow` (and `pandas`) would use, but isn't required (e.g. `polars`)."""

    __slots__ = ()


@overload
def can_eager(
    obj: EagerClasses[EagerDF, S, EagerE, EagerSC],
) -> TypeIs[EagerClasses[EagerDF, S, EagerE, EagerSC]]: ...
@overload
def can_eager(obj: EagerClasses[DF, S, E, SC]) -> TypeIs[EagerClasses[DF, S, E, SC]]: ...
def can_eager(
    obj: EagerClasses[DF, S, E, SC] | EagerClasses[EagerDF, S, EagerE, EagerSC] | Any,
) -> (
    TypeIs[EagerClasses[DF, S, E, SC]] | TypeIs[EagerClasses[EagerDF, S, EagerE, EagerSC]]
):
    return hasattrs_static(obj, "_dataframe", "_series")


def can_lazy(obj: LazyClasses[LF, PE, E, SC] | Any) -> TypeIs[LazyClasses[LF, PE, E, SC]]:
    return hasattrs_static(obj, "_lazyframe", "_evaluator")


def can_v1(obj: HasV1[C1] | Any) -> TypeIs[HasV1[C1]]:
    return hasattrs_static(obj, "v1")


def can_v2(obj: HasV2[C2] | Any) -> TypeIs[HasV2[C2]]:
    return hasattrs_static(obj, "v2")


# NOTE: As `ScalarT_co` has a default, it must be listed at the end of the type parameters
# The important addition to this level is `HasV1[...]`, so  `...` is listed first
# fmt: off
class EagerClassesV1(EagerClasses[DF, S, E, SC], HasV1[C1], Protocol[C1, DF, S, E, SC]):
    __slots__ = ()

class EagerClassesV2(EagerClasses[DF, S, E, SC], HasV2[C2], Protocol[C2, DF, S, E, SC]):
    __slots__ = ()

class EagerClassesVAll(EagerClasses[DF, S, E, SC], HasVAll[C1, C2], Protocol[C1, C2, DF, S, E, SC]):
    __slots__ = ()

class LazyClassesV1(LazyClasses[LF, PE, E, SC], HasV1[C1], Protocol[C1, LF, PE, E, SC]):
    __slots__ = ()

class LazyClassesV2(LazyClasses[LF, PE, E, SC], HasV2[C2], Protocol[C2, LF, PE, E, SC]):
    __slots__ = ()

class LazyClassesVAll(LazyClasses[LF, PE, E, SC], HasVAll[C1, C2], Protocol[C1, C2, LF, PE, E, SC]):
    __slots__ = ()

class HybridClassesV1(HybridClasses[DF, S, LF, PE, E, SC], HasV1[C1], Protocol[C1, DF, S, LF, PE, E, SC]):
    __slots__ = ()

class HybridClassesV2(HybridClasses[DF, S, LF, PE, E, SC], HasV2[C2], Protocol[C2, DF, S, LF, PE, E, SC]):
    __slots__ = ()

class HybridClassesVAll(HybridClasses[DF, S, LF, PE, E, SC], HasVAll[C1, C2], Protocol[C1, C2, DF, S, LF, PE, E, SC]):
    __slots__ = ()

class EagerImplClassesVAll(EagerImplClasses[EagerDF, S, EagerE, EagerSC], HasVAll[EagerImplC1, EagerImplC2], Protocol[EagerImplC1, EagerImplC2, EagerDF, S, EagerE, EagerSC]):
    __slots__ = ()
# fmt: on
