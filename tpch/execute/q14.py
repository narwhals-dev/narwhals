from __future__ import annotations

from queries import q14

from . import IO_FUNCS
from . import line_item
from . import part

tool = "pandas[pyarrow]"
fn = IO_FUNCS[tool]
print(q14.query(fn(line_item), fn(part)))

tool = "polars[lazy]"
fn = IO_FUNCS[tool]
print(q14.query(fn(line_item), fn(part)).collect())

tool = "pyarrow"
fn = IO_FUNCS[tool]
print(q14.query(fn(line_item), fn(part)))

tool = "dask"
fn = IO_FUNCS[tool]
print(q14.query(fn(line_item), fn(part)).compute())
