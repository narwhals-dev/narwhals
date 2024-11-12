from __future__ import annotations

from queries import q11

from . import IO_FUNCS
from . import nation
from . import partsupp
from . import supplier

tool = "pandas[pyarrow]"
fn = IO_FUNCS[tool]
print(q11.query(fn(nation), fn(partsupp), fn(supplier)))

tool = "polars[lazy]"
fn = IO_FUNCS[tool]
print(q11.query(fn(nation), fn(partsupp), fn(supplier)).collect())

tool = "pyarrow"
fn = IO_FUNCS[tool]
print(q11.query(fn(nation), fn(partsupp), fn(supplier)))
