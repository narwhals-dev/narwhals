from __future__ import annotations

from queries import q5

from . import IO_FUNCS
from . import customer
from . import line_item
from . import nation
from . import orders
from . import region
from . import supplier

tool = "pandas[pyarrow]"
fn = IO_FUNCS[tool]
print(
    q5.query(
        fn(region), fn(nation), fn(customer), fn(line_item), fn(orders), fn(supplier)
    )
)

tool = "polars[lazy]"
fn = IO_FUNCS[tool]
print(
    q5.query(
        fn(region), fn(nation), fn(customer), fn(line_item), fn(orders), fn(supplier)
    ).collect()
)

tool = "pyarrow"
fn = IO_FUNCS[tool]
print(
    q5.query(
        fn(region), fn(nation), fn(customer), fn(line_item), fn(orders), fn(supplier)
    )
)
