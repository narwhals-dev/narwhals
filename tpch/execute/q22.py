from __future__ import annotations

from queries import q22

from . import IO_FUNCS
from . import customer
from . import orders

tool = "pandas[pyarrow]"
fn = IO_FUNCS[tool]
print(q22.query(fn(customer), fn(orders)))

tool = "polars[lazy]"
fn = IO_FUNCS[tool]
print(q22.query(fn(customer), fn(orders)).collect())

tool = "pyarrow"
fn = IO_FUNCS[tool]
print(q22.query(fn(customer), fn(orders)))
