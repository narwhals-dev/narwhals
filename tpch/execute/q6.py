from __future__ import annotations

from queries import q6

from . import IO_FUNCS
from . import lineitem

tool = "pandas[pyarrow]"
fn = IO_FUNCS[tool]
print(q6.query(fn(lineitem)))

tool = "polars[lazy]"
fn = IO_FUNCS[tool]
print(q6.query(fn(lineitem)).collect())

tool = "pyarrow"
fn = IO_FUNCS[tool]
print(q6.query(fn(lineitem)))
