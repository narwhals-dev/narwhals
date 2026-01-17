"""Functions about functions."""

from __future__ import annotations

import typing as t
from typing import TYPE_CHECKING

import pyarrow as pa  # ignore-banned-import
import pyarrow.compute as pc  # ignore-banned-import

if TYPE_CHECKING:
    from typing_extensions import LiteralString, TypeAlias

    from narwhals._plan.arrow.typing import ArrowAny
    from narwhals.typing import PythonLiteral


__all__ = ["call"]

Incomplete: TypeAlias = t.Any

_PackComputeArgsElement: TypeAlias = (
    "PythonLiteral | ArrowAny | pa.RecordBatch | pa.Table"
)
"""[`_pack_compute_args`] covers every possible input types to a `pyarrow.compute` function.

This version just excludes `np.ndarray` (*for now*).

[`_pack_compute_args`]: https://github.com/apache/arrow/blob/29586f4d28c50a4344f14a78dc7e091ab635fa72/python/pyarrow/_compute.pyx#L488-L520
"""


def call(
    name: LiteralString,
    *args: _PackComputeArgsElement,
    options: pc.FunctionOptions | None = None,
) -> Incomplete:
    """Call a [`pyarrow.compute`] function by name.

    Escape hatch to use when typing falls apart.

    Arguments:
        name: Name of the function to call.
        *args: Arguments to the function.
        options: A [`pc.FunctionOptions`] instance to pass to the function.

    [`pyarrow.compute`]: https://arrow.apache.org/docs/dev/python/generated/pyarrow.compute.call_function.html
    [`pc.FunctionOptions`]: https://arrow.apache.org/docs/dev/python/api/compute.html#compute-options
    """
    call_function: Incomplete = pc.call_function
    return call_function(name, args, options=options)
