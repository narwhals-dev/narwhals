"""The class-accessing parts of `*Namespace`.

## Notes
- Separates the operations from navigation
- Exposed in the `Plugin`-level as `__narwhals_classes__`
- Intended to be used in a similar way for the implementation
    - But not by accessing through the plugin
- A stepping stone to the `Package` idea.
    - Everything is covariant already
    - This object can be put anywhere
    - Still uses the current names, so less noise
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Protocol, TypeVar

from narwhals._plan.compliant.typing import (
    DataFrameT_co,
    EagerDataFrameT_co,
    EagerExprT_co,
    EagerScalarT_co,
    ExprT_co,
    LazyFrameT_co,
    ScalarT_co,
    SeriesT_co,
)

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

    from narwhals._plan.plans.visitors import ResolvedToCompliant
    from narwhals._utils import Version


EagerClassesAny: TypeAlias = "EagerClasses[Any, Any, Any, Any]"
EagerImplClassesAny: TypeAlias = "EagerImplClasses[Any, Any, Any, Any]"
LazyClassesAny: TypeAlias = "LazyClasses[Any, Any, Any, Any]"
ClassesAny: TypeAlias = "EagerClassesAny | LazyClassesAny"
PlanEvaluatorAny: TypeAlias = "ResolvedToCompliant[Any]"
PlanEvaluatorT_co = TypeVar("PlanEvaluatorT_co", bound=PlanEvaluatorAny, covariant=True)
EagerClassesT_co = TypeVar("EagerClassesT_co", bound=EagerClassesAny, covariant=True)
EagerImplClassesT_co = TypeVar(
    "EagerImplClassesT_co", bound=EagerImplClassesAny, covariant=True
)
LazyClassesT_co = TypeVar("LazyClassesT_co", bound=LazyClassesAny, covariant=True)
ClassesT_co = TypeVar("ClassesT_co", bound=ClassesAny, covariant=True)


class CompliantClasses(Protocol[ExprT_co, ScalarT_co]):
    """The class-accessing parts of `*Namespace`.

    ## Notes
    - `Implementation` is available in the plugin and in each class, no need for here too.
    - `__narwhals_expr_prepare__` is still needed for disambiguating `Expr`, `Scalar` and `Namespace`
        - Although the adding state part is not used any more
    """

    __slots__ = ()
    version: ClassVar[Version]

    @property
    def _expr(self) -> type[ExprT_co]: ...
    @property
    def _scalar(self) -> type[ScalarT_co]: ...
    def __narwhals_expr_prepare__(self) -> ExprT_co:
        tp = self._expr
        return tp.__new__(tp)


class EagerClasses(
    CompliantClasses[ExprT_co, ScalarT_co],
    Protocol[DataFrameT_co, SeriesT_co, ExprT_co, ScalarT_co],
):
    __slots__ = ()

    @property
    def _dataframe(self) -> type[DataFrameT_co]: ...
    @property
    def _series(self) -> type[SeriesT_co]: ...


class EagerImplClasses(
    EagerClasses[EagerDataFrameT_co, SeriesT_co, EagerExprT_co, EagerScalarT_co],
    Protocol[EagerDataFrameT_co, SeriesT_co, EagerExprT_co, EagerScalarT_co],
):
    __slots__ = ()


class LazyClasses(
    CompliantClasses[ExprT_co, ScalarT_co],
    Protocol[LazyFrameT_co, PlanEvaluatorT_co, ExprT_co, ScalarT_co],
):
    __slots__ = ()

    @property
    def _lazyframe(self) -> type[LazyFrameT_co]: ...
    @property
    def _evaluator(self) -> type[PlanEvaluatorT_co]: ...


class HasClasses(Protocol[ClassesT_co]):
    """An object or module that defines `__narwhals_classes__`.

    Provides access to all compliant-level implementation classes that have been defined.
    """

    __slots__ = ()

    @property
    def __narwhals_classes__(self) -> ClassesT_co: ...
