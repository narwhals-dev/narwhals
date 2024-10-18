from __future__ import annotations

from queries import q17

from . import IO_FUNCS
from . import lineitem
from . import part

tool = "pandas[pyarrow]"
fn = IO_FUNCS[tool]
print(q17.query(fn(lineitem), fn(part)))

tool = "polars[lazy]"
fn = IO_FUNCS[tool]
print(q17.query(fn(lineitem), fn(part)).collect())

tool = "pyarrow"
fn = IO_FUNCS[tool]
print(q17.query(fn(lineitem), fn(part)))

tool = "dask"
fn = IO_FUNCS[tool]
print(q17.query(fn(lineitem), fn(part)).compute())
