from __future__ import annotations

from queries import q7

from . import IO_FUNCS
from . import customer
from . import lineitem
from . import nation
from . import orders
from . import supplier

tool = "pandas[pyarrow]"
fn = IO_FUNCS[tool]
print(q7.query(fn(nation), fn(customer), fn(lineitem), fn(orders), fn(supplier)))

tool = "polars[lazy]"
fn = IO_FUNCS[tool]
print(
    q7.query(fn(nation), fn(customer), fn(lineitem), fn(orders), fn(supplier)).collect()
)

tool = "pyarrow"
fn = IO_FUNCS[tool]
print(q7.query(fn(nation), fn(customer), fn(lineitem), fn(orders), fn(supplier)))

tool = "dask"
fn = IO_FUNCS[tool]
print(
    q7.query(fn(nation), fn(customer), fn(lineitem), fn(orders), fn(supplier)).compute()
)
