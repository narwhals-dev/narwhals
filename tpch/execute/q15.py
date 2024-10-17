from __future__ import annotations

from queries import q15

from . import IO_FUNCS
from . import lineitem
from . import supplier

tool = "pandas[pyarrow]"
fn = IO_FUNCS[tool]
print(q15.query(fn(lineitem), fn(supplier)))

tool = "polars[lazy]"
fn = IO_FUNCS[tool]
print(q15.query(fn(lineitem), fn(supplier)).collect())

tool = "pyarrow"
fn = IO_FUNCS[tool]
print(q15.query(fn(lineitem), fn(supplier)))

tool = "dask"
fn = IO_FUNCS[tool]
print(q15.query(fn(lineitem), fn(supplier)).compute())
