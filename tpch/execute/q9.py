from __future__ import annotations

from queries import q9

from . import IO_FUNCS
from . import lineitem
from . import nation
from . import orders
from . import part
from . import partsupp
from . import supplier

tool = "pandas[pyarrow]"
fn = IO_FUNCS[tool]
print(
    q9.query(fn(part), fn(partsupp), fn(nation), fn(lineitem), fn(orders), fn(supplier))
)

tool = "polars[lazy]"
fn = IO_FUNCS[tool]
print(
    q9.query(
        fn(part), fn(partsupp), fn(nation), fn(lineitem), fn(orders), fn(supplier)
    ).collect()
)

tool = "pyarrow"
fn = IO_FUNCS[tool]
print(
    q9.query(fn(part), fn(partsupp), fn(nation), fn(lineitem), fn(orders), fn(supplier))
)
