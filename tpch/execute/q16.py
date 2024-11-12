from __future__ import annotations

from queries import q16

from . import IO_FUNCS
from . import part
from . import partsupp
from . import supplier

tool = "pandas[pyarrow]"
fn = IO_FUNCS[tool]
print(q16.query(fn(part), fn(partsupp), fn(supplier)))

tool = "polars[lazy]"
fn = IO_FUNCS[tool]
print(q16.query(fn(part), fn(partsupp), fn(supplier)).collect())

tool = "pyarrow"
fn = IO_FUNCS[tool]
print(q16.query(fn(part), fn(partsupp), fn(supplier)))
