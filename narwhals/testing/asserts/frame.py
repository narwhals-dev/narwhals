from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from narwhals.typing import IntoFrameT


def assert_frame_equal(
    left: IntoFrameT,
    right: IntoFrameT,
    *,
    check_row_order: bool = True,
    check_column_order: bool = True,
    check_dtypes: bool = True,
    check_exact: bool = False,
    rel_tol: float = 1e-05,
    abs_tol: float = 1e-08,
    categorical_as_str: bool = False,
) -> None:
    msg = "TODO"  # pragma: no cover
    raise NotImplementedError(msg)
