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

from typing import TYPE_CHECKING, Any, ClassVar, Protocol, TypeVar

from narwhals._plan.common import hasattrs_static
from narwhals._plan.compliant.typing import (
    # NOTE: The names are simply too long!
    DataFrameAny,
    DataFrameT_co as DF,
    EagerDataFrameAny,
    EagerDataFrameT_co,
    EagerExprAny,
    EagerExprT_co,
    EagerScalarAny,
    EagerScalarT_co,
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


EagerClassesAny: TypeAlias = (
    "EagerClasses[DataFrameAny, SeriesAny, ExprAny, ExprAny | ScalarAny]"
)
LazyClassesAny: TypeAlias = (
    "LazyClasses[LazyFrameAny, PlanEvaluatorAny, ExprAny, ExprAny | ScalarAny]"
)
HybridClassesAny: TypeAlias = "HybridClasses[DataFrameAny, SeriesAny, LazyFrameAny, PlanEvaluatorAny, ExprAny, ExprAny | ScalarAny]"

ClassesAny: TypeAlias = "EagerClassesAny | LazyClassesAny | HybridClassesAny"
"""The type of either `__narwhals_classes__` or `__narwhals_classes__.v*`.

Can provide eager, lazy or a combination of the two.
"""

# NOTE: The type of `__narwhals_classes__`, which defines a `v1` or `v2` property
EagerClassesV1Any: TypeAlias = "EagerClassesV1[EagerClassesAny, DataFrameAny, SeriesAny, ExprAny, ExprAny | ScalarAny]"
EagerClassesV2Any: TypeAlias = "EagerClassesV2[EagerClassesAny, DataFrameAny, SeriesAny, ExprAny, ExprAny | ScalarAny]"
LazyClassesV1Any: TypeAlias = "LazyClassesV1[LazyClassesAny, LazyFrameAny, PlanEvaluatorAny, ExprAny, ExprAny | ScalarAny]"
LazyClassesV2Any: TypeAlias = "LazyClassesV2[LazyClassesAny, LazyFrameAny, PlanEvaluatorAny, ExprAny, ExprAny | ScalarAny]"
ClassesV1Any: TypeAlias = "EagerClassesV1Any | LazyClassesV1Any"
"""The type of `__narwhals_classes__` which defines a `v1` property."""
ClassesV2Any: TypeAlias = "EagerClassesV2Any | LazyClassesV2Any"
"""The type of `__narwhals_classes__` which defines a `v2` property."""
ClassesVAny: TypeAlias = "ClassesV1Any | ClassesV2Any"
"""The type of `__narwhals_classes__` which defines at least one versioned property."""

ClassesT_co = TypeVar("ClassesT_co", bound=ClassesAny, covariant=True)
"""Covariant TypeVar for the `__narwhals_classes__` property."""
ClassesV1T_co = TypeVar("ClassesV1T_co", bound=ClassesAny, covariant=True)
"""Covariant TypeVar for the `__narwhals_classes__.v1` property."""
ClassesV2T_co = TypeVar("ClassesV2T_co", bound=ClassesAny, covariant=True)
"""Covariant TypeVar for the `__narwhals_classes__.v2` property."""
HasClassesV1T_co = TypeVar("HasClassesV1T_co", bound="ClassesV1Any", covariant=True)
HasClassesV2T_co = TypeVar("HasClassesV2T_co", bound="ClassesV2Any", covariant=True)
# Not used yet
EagerClassesT_co = TypeVar("EagerClassesT_co", bound=EagerClassesAny, covariant=True)
LazyClassesT_co = TypeVar("LazyClassesT_co", bound=LazyClassesAny, covariant=True)
HybridClassesT_co = TypeVar("HybridClassesT_co", bound=HybridClassesAny, covariant=True)


class HasClasses(Protocol[ClassesT_co]):
    """An object or module that defines `__narwhals_classes__`."""

    __slots__ = ()

    @property
    def __narwhals_classes__(self) -> ClassesT_co:
        """The backend's compliant-level types."""
        ...


class HasV1(Protocol[ClassesV1T_co]):
    """An object that defines `v1`."""

    __slots__ = ()

    @property
    def v1(self) -> ClassesV1T_co:
        """The classes to use with `Version.V1`."""
        ...


class HasV2(Protocol[ClassesV2T_co]):
    """An object that defines `v2`."""

    __slots__ = ()

    @property
    def v2(self) -> ClassesV2T_co:
        """The classes to use with `Version.V2`."""
        ...


HasClassesV1 = HasClasses[HasClassesV1T_co]
"""Extends `__narwhals_classes__` with a `v1` property."""
HasClassesV2 = HasClasses[HasClassesV2T_co]
"""Extends `__narwhals_classes__` with a `v2` property."""


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
    __slots__ = ()


def can_eager(
    obj: EagerClasses[DF, S, E, SC] | Any,
) -> TypeIs[EagerClasses[DF, S, E, SC]]:
    return hasattrs_static(obj, "_dataframe", "_series")


def can_lazy(obj: LazyClasses[LF, PE, E, SC] | Any) -> TypeIs[LazyClasses[LF, PE, E, SC]]:
    return hasattrs_static(obj, "_lazyframe", "_evaluator")


def can_v1(obj: HasV1[ClassesV1T_co] | Any) -> TypeIs[HasV1[ClassesV1T_co]]:
    return hasattrs_static(obj, "v1")


def can_v2(obj: HasV2[ClassesV2T_co] | Any) -> TypeIs[HasV2[ClassesV2T_co]]:
    return hasattrs_static(obj, "v2")


# NOTE: As `ScalarT_co` has a default, it must be listed at the end of the type parameters
# The important addition to this level is `HasV1[...]`, so  `...` is listed first
# fmt: off
class EagerClassesV1(EagerClasses[DF, S, E, SC], HasV1[ClassesV1T_co], Protocol[ClassesV1T_co, DF, S, E, SC]):
    __slots__ = ()

class EagerClassesV2(EagerClasses[DF, S, E, SC], HasV2[ClassesV2T_co], Protocol[ClassesV2T_co, DF, S, E, SC]):
    __slots__ = ()

class LazyClassesV1(LazyClasses[LF, PE, E, SC], HasV1[ClassesV1T_co], Protocol[ClassesV1T_co, LF, PE, E, SC]):
    __slots__ = ()

class LazyClassesV2(LazyClasses[LF, PE, E, SC], HasV2[ClassesV2T_co], Protocol[ClassesV2T_co, LF, PE, E, SC]):
    __slots__ = ()
# fmt: on

EagerImplClasses = EagerClasses[EagerDataFrameT_co, S, EagerExprT_co, EagerScalarT_co]
"""This is what `pyarrow` (and `pandas`) would use, but isn't required (e.g. `polars`)."""
EagerImplClassesAny: TypeAlias = "EagerImplClasses[EagerDataFrameAny, SeriesAny, EagerExprAny, EagerExprAny | EagerScalarAny]"
EagerImplClassesT_co = TypeVar(
    "EagerImplClassesT_co", bound=EagerImplClassesAny, covariant=True
)
